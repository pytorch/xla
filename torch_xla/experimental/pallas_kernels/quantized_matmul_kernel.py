import functools
from typing import cast
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


NUM_LANES = 128
NUM_SUBLANES = 8

def _broadcast_to_shape(arr, shape, broadcast_dim):
  if arr.shape == shape:
    return arr
  assert len(arr.shape) == len(shape)
  assert len(shape) == 2
  assert broadcast_dim in [0, 1]

  non_broadcast_dim = 0 if broadcast_dim == 1 else 1
  assert arr.shape[non_broadcast_dim] == shape[non_broadcast_dim]
  assert shape[broadcast_dim] % arr.shape[broadcast_dim] == 0
  # no-op concatenation.
  return jnp.concatenate(
      [arr for _ in range(shape[broadcast_dim] // arr.shape[broadcast_dim])], axis=broadcast_dim
  )

def quantize_array_kernel(x_ref, out_ref, scale_ref, *, quant_dtype, quant_nbits):
  x = x_ref[...].astype(jnp.float32)
  max_val = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
  int_min = -2**(quant_nbits - 1)
  int_max = 2**(quant_nbits - 1) - 1
  scale = max_val / int_max
  x_int = jnp.clip(jnp.round((x / scale).astype(jnp.float32)), int_min, int_max).astype(quant_dtype)

  out_ref[...] = x_int
  scale_ref[...] = scale.astype(x_ref.dtype)

def quantize_array(
    x: jax.Array,  # [m, n]
    vmem_limit_bytes,
):
  quant_dtype = jnp.int8
  quant_nbits = 8
  m, n = x.shape
  
  def within_vmem_limit(m_block_size, n, orig_dtype, vmem_limit_bytes):
    return 2*(2*m_block_size * n + m_block_size)*orig_dtype.itemsize <= vmem_limit_bytes
  
  def find_m_block_size(x, vmem_limit_bytes):
    m, n = x.shape
    for m_block_size in [1024, 512, 256]:
      if m % m_block_size == 0 and within_vmem_limit(m_block_size, n, x.dtype, vmem_limit_bytes):
        return m_block_size
    if not within_vmem_limit(m, n, x.dtype, vmem_limit_bytes):
      raise ValueError(f"Cannot find m_block_size for {x.shape=} with vmem_limit_bytes={vmem_limit_bytes}")
    return m
  bm = find_m_block_size(x, vmem_limit_bytes)

  input_spec = pl.BlockSpec((bm, n), lambda i: (i, 0))
  scale_spec = pl.BlockSpec((bm, 1), lambda i: (i, 0))
  output_shape = jax.ShapeDtypeStruct((m, n), dtype=quant_dtype)
  scale_shape = jax.ShapeDtypeStruct((m, 1), dtype=x.dtype)
  kernel = pl.pallas_call(
      functools.partial(
          quantize_array_kernel,
          quant_dtype=quant_dtype,
          quant_nbits=quant_nbits,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              input_spec,
          ],
          out_specs=[
              input_spec,
              scale_spec,
          ],
          grid=(m // bm,),
      ),
      out_shape=(output_shape, scale_shape),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel",),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )

  return kernel(x)

def w8a8_matmul_kernel(x_ref, x_scalar_ref, w_ref, w_scalar_ref, out_ref, acc_ref, *, x_orig_dtype, batch_block_size, out_block_size, in_block_size):
  bs_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  nsteps = pl.num_programs(2)
  
  @pl.when(in_idx == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  x = x_ref[...]
  w = w_ref[...]
  acc_ref[...] += jax.lax.dot_general(
      x,
      w,
      (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.int32,
  )

  @pl.when(in_idx == nsteps - 1)
  def _():
    acc = acc_ref[...]  # [batch_block_size, out_block_size]
    w_scalar = w_scalar_ref[...]  # [NUM_SUBLANES, out_block_size]
    acc *= _broadcast_to_shape(w_scalar, acc.shape, broadcast_dim=0)
    x_scalar = x_scalar_ref[...]  # [batch_block_size, NUM_LANES]
    acc *= _broadcast_to_shape(x_scalar, acc.shape, broadcast_dim=1)
    out_ref[...] = acc.astype(x_orig_dtype)

def w8a8_matmul(
    x: jax.Array,  # [bs, n_input_features]
    x_scalar: jax.Array,  # [bs, 1]
    w: jax.Array,  # [n_output_features, n_input_features]
    w_scalar: jax.Array,  # [1, n_output_features]
    x_orig_dtype,
    batch_block_size,
    out_block_size,
    in_block_size,
    vmem_limit_bytes,
):
  assert x.dtype == jnp.int8, f"x.dtype ({x.dtype}) must be int8"
  assert w.dtype == jnp.int8, f"w.dtype ({w.dtype}) must be int8"

  bs, n_input_features = x.shape
  n_output_features, _ = w.shape
  x_scalar = jnp.broadcast_to(x_scalar, (x_scalar.shape[0], NUM_LANES))
  w_scalar = jnp.broadcast_to(w_scalar, (NUM_SUBLANES, w_scalar.shape[1]))

  acc_dtype = jnp.int32
  kernel = pl.pallas_call(
      functools.partial(
          w8a8_matmul_kernel,
          x_orig_dtype=x_orig_dtype,
          batch_block_size=batch_block_size,
          out_block_size=out_block_size,
          in_block_size=in_block_size),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i:
                           (b, i)),  # x
              pl.BlockSpec((batch_block_size, NUM_LANES), lambda b, o, i:
                           (b, 0)),  # x_scalar
              pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i:
                           (o, i)),  # w
              pl.BlockSpec((NUM_SUBLANES, out_block_size), lambda b, o, i:
                           (0, o)),  # w_scalar
          ],
          out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                 lambda b, o, i: (b, o)),
          scratch_shapes=[
              pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
          ],
          grid=(bs // batch_block_size,
                n_output_features // out_block_size,
                n_input_features // in_block_size),
      ),
      out_shape=jax.ShapeDtypeStruct((bs, n_output_features), x_orig_dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )
  return kernel(x, x_scalar, w, w_scalar)

@functools.partial(
    jax.jit,
    static_argnames=[
        'quantize_activation',
        'batch_block_size',
        'out_block_size',
        'in_block_size',
        'vmem_limit_bytes',
    ])
def quantized_matmul_int8(
    x: jax.Array,  # [bs, n_input_features]
    w: jax.Array,  # [n_output_features, n_input_features]
    scalar: jax.Array,  # [n_output_features]
    zero_point: jax.Array | None = None,
    quant_block_size: jax.Array | None = None,  # i32[1]
    quantize_activation: bool = False,
    *,
    # All 3 block sizes, if provided, have to be multiples of 128 because they are used as the minormost dimension in the block.
    batch_block_size: int | None = None,
    out_block_size: int | None = None,
    in_block_size: int | None = None,
    vmem_limit_bytes: int | None = 64 * 1024 * 1024,
):
  assert zero_point is None, "Not implemented: zero_point is not supported."
  assert quant_block_size is None, "Not implemented: quant_block_size is not supported."

  orig_bs, orig_in_features = x.shape
  orig_out_features, _ = w.shape
  if batch_block_size is None or out_block_size is None or in_block_size is None:
    batch_block_size, out_block_size, in_block_size = get_tuned_block_sizes(TUNED_BLOCK_SIZES, orig_bs, orig_out_features, orig_in_features, jnp.dtype(x.dtype).name, quantize_activation)
    
  orig_batch_block_size = batch_block_size
  if orig_bs < 128:
    batch_block_size = orig_bs

  assert x.shape[1] == w.shape[
      1], f"x.shape[1] ({x.shape[1]}) must be equal to w.shape[1] ({w.shape[1]})"
  assert w.shape[0] == scalar.shape[
      0], f"w.shape[0] ({w.shape[0]}) must be equal to scalar.shape[0] ({scalar.shape[0]})"
  assert x.shape[
      0] % batch_block_size == 0, f"x.shape[0] ({x.shape[0]}) must be a multiple of block size ({batch_block_size})"
  assert w.shape[
      0] % out_block_size == 0, f"w.shape[0] ({w.shape[0]}) must be a multiple of block size ({out_block_size})"
  assert x.shape[
      1] % in_block_size == 0, f"x.shape[1] ({x.shape[1]}) must be a multiple of block size ({in_block_size})"

  x_orig_dtype = x.dtype
  x_int, x_scalar = quantize_array(x, vmem_limit_bytes)
  assert x_int.dtype == jnp.int8, f"x_int.dtype ({x_int.dtype}) must be int8"
  assert w.dtype == jnp.int8, f"w.dtype ({w.dtype}) must be int8"

  if scalar.dtype != jnp.float32:
    scalar = scalar.astype(jnp.float32)
  scalar = jnp.expand_dims(scalar, axis=0)  # [1, n_output_features]

  kernel_name = f'quantized_matmul_int8_{orig_batch_block_size}_{out_block_size}_{in_block_size}'
  # The named_scope is used for autotune. Different block sizes only impact the
  # pallas_call.
  with jax.named_scope(kernel_name):
    out = w8a8_matmul(x_int, x_scalar, w, scalar, x_orig_dtype, batch_block_size, out_block_size, in_block_size, vmem_limit_bytes)
  return out


# Below are tuned block sizes.

# key:
#    - tpu_version
#    - batch_size
#    - n_output_features
#    - n_input_features
#    - activation_dtype
#    - quantize_activation
# value:
#    - batch_block_size
#    - out_block_size
#    - in_block_size
TUNED_BLOCK_SIZES = {
    (6, 1024, 1280, 8192, 'bfloat16', True): (512, 1280, 8192),
    (6, 1024, 28672, 4096, 'bfloat16', True): (1024, 1792, 4096),
    (6, 1024, 4096, 14336, 'bfloat16', True): (1024, 256, 14336),
    (6, 1024, 4096, 4096, 'bfloat16', True): (1024, 512, 4096),
    (6, 1024, 6144, 4096, 'bfloat16', True): (1024, 768, 4096),
    (6, 1024, 7168, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 1024, 8192, 1024, 'bfloat16', True): (1024, 4096, 1024),
    (6, 1024, 8192, 3584, 'bfloat16', True): (1024, 1024, 3584),
    (6, 128, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 128, 28672, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 128, 4096, 14336, 'bfloat16', True): (128, 256, 14336),
    (6, 128, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 128, 6144, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 128, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 128, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 128, 8192, 3584, 'bfloat16', True): (128, 512, 3584),
    (6, 16, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 16, 28672, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 16, 4096, 14336, 'bfloat16', True): (128, 1024, 2048),
    (6, 16, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 16, 6144, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 16, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 16, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 16, 8192, 3584, 'bfloat16', True): (128, 4096, 896),
    (6, 2048, 1280, 8192, 'bfloat16', True): (512, 1280, 8192),
    (6, 2048, 28672, 4096, 'bfloat16', True): (2048, 1024, 4096),
    (6, 2048, 4096, 14336, 'bfloat16', True): (2048, 512, 14336),
    (6, 2048, 4096, 4096, 'bfloat16', True): (256, 4096, 4096),
    (6, 2048, 6144, 4096, 'bfloat16', True): (2048, 256, 4096),
    (6, 2048, 7168, 8192, 'bfloat16', True): (2048, 512, 8192),
    (6, 2048, 8192, 1024, 'bfloat16', True): (256, 8192, 1024),
    (6, 2048, 8192, 3584, 'bfloat16', True): (1024, 512, 3584),
    (6, 256, 1280, 8192, 'bfloat16', True): (256, 256, 8192),
    (6, 256, 28672, 4096, 'bfloat16', True): (256, 3584, 2048),
    (6, 256, 4096, 14336, 'bfloat16', True): (256, 1024, 3584),
    (6, 256, 4096, 4096, 'bfloat16', True): (256, 512, 4096),
    (6, 256, 6144, 4096, 'bfloat16', True): (256, 768, 4096),
    (6, 256, 7168, 8192, 'bfloat16', True): (256, 512, 8192),
    (6, 256, 8192, 1024, 'bfloat16', True): (256, 2048, 1024),
    (6, 256, 8192, 3584, 'bfloat16', True): (256, 1024, 3584),
    (6, 32, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 32, 28672, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 32, 4096, 14336, 'bfloat16', True): (128, 1024, 3584),
    (6, 32, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 32, 6144, 4096, 'bfloat16', True): (128, 1536, 2048),
    (6, 32, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 32, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 32, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    (6, 512, 1280, 8192, 'bfloat16', True): (512, 256, 8192),
    (6, 512, 28672, 4096, 'bfloat16', True): (512, 4096, 4096),
    (6, 512, 4096, 14336, 'bfloat16', True): (512, 256, 14336),
    (6, 512, 4096, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 6144, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 7168, 8192, 'bfloat16', True): (512, 512, 8192),
    (6, 512, 8192, 1024, 'bfloat16', True): (512, 4096, 1024),
    (6, 512, 8192, 3584, 'bfloat16', True): (512, 2048, 3584),
    (6, 64, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 64, 28672, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 64, 4096, 14336, 'bfloat16', True): (128, 512, 7168),
    (6, 64, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 64, 6144, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 64, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 64, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 64, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
}


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[:-len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_tuned_block_sizes(batch_size, n_output_features, n_input_features,
                          activation_dtype, quantize_activation):
  """
    Retrieve the tuned block sizes for the given parameters.
    
    Args:
        batch_size (int): The batch size.
        n_output_features (int): The number of output features.
        n_input_features (int): The number of input features.
        activation_dtype (str): The data type of the activation ('bfloat16' or 'float32').
        quantize_activation (bool): Whether to quantize the activation.
        
    Returns:
        tuple: A tuple containing the batch_block_size, out_block_size, and in_block_size.
    """
  key = (get_tpu_version(), batch_size, n_output_features, n_input_features,
         activation_dtype, quantize_activation)
  return TUNED_BLOCK_SIZES.get(key, (128, 128, 128))
