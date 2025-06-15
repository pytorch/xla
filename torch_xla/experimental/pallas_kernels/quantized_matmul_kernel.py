import functools
from typing import cast
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max_val: jax.Array,  # [1, bs_block_size]
):
  n_bits = 8
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = (x_abs_max_val / int_max).T  # [bs_block_size, 1]
  # Need to explicitly cast to f32 because Mosaic can't directly jnp.round a
  # bf16 array.
  # It seems x/0 in Pallas generates inf/-inf instead of an exception.
  x_int = jnp.clip(
      jnp.round((x / scale).astype(jnp.float32)), int_min,
      int_max).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)


def matmul_kernel(
    x_ref,  # (batch_block_size, in_block_size)
    w_ref,  # (out_block_size, in_block_size)
    scalar_ref,  # (1, out_block_size)
    x_abs_max_val,  # (1, batch_block_size)
    out_ref,  # (batch_block_size, out_block_size)
    acc_ref,  # (batch_block_size, out_block_size)
    *,
    quantize_activation,
    batch_block_size,
    out_block_size,
    in_block_size,
):
  bs_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  nsteps = pl.num_programs(2)
  x_ref_dtype = x_ref.dtype
  assert x_ref.shape == (batch_block_size,
                         in_block_size), "x_ref shape is not correct"
  assert w_ref.shape == (out_block_size,
                         in_block_size), "w_ref shape is not correct"
  assert scalar_ref.shape == (1,
                              out_block_size), "scalar_ref shape is not correct"
  assert x_abs_max_val.shape == (
      1, batch_block_size), "x_max_val shape is not correct"
  assert out_ref.shape == (batch_block_size,
                           out_block_size), "out_ref shape is not correct"
  assert acc_ref.shape == (batch_block_size,
                           out_block_size), "acc_ref shape is not correct"

  @pl.when(in_idx == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  if quantize_activation:
    x, x_scale = _quantize_array(x_ref[...], x_abs_max_val[...])
    acc_ref[...] += jax.lax.dot_general(
        x,
        w_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
  else:
    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        (((1,), (1,)), ((), ())),
    )

  @pl.when(in_idx == nsteps - 1)
  def _():
    acc = acc_ref[...]
    scalar = scalar_ref[...]
    acc *= scalar
    if quantize_activation:
      acc *= x_scale
    out_ref[...] = acc.astype(x_ref_dtype)


def _next_multiple(x, multiple):
  return ((x + multiple - 1) // multiple) * multiple


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

  # x_max_val cannot be [bs, 128] where 128 is the minormost dimension of the vreg because it'll be costly to store
  # [bs_block_size, 128] in VMEM ([bs_balock_size, 1:] will be padding).
  # We need the global max values to be computed before the kernel.
  x_abs_max_val = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
  x_abs_max_val = jnp.expand_dims(x_abs_max_val, axis=0)  # [1, bs]
  assert x_abs_max_val.shape == (1, x.shape[0])

  orig_bs, orig_in_features = x.shape
  orig_out_features, _ = w.shape
  if batch_block_size is None or out_block_size is None or in_block_size is None:
    batch_block_size, out_block_size, in_block_size = get_tuned_block_sizes(
        orig_bs, orig_out_features, orig_in_features,
        jnp.dtype(x.dtype).name, quantize_activation)

  padded_bs = _next_multiple(orig_bs, batch_block_size)
  if orig_bs < padded_bs:
    x = jnp.pad(x, ((0, padded_bs - orig_bs), (0, 0)))
    x_abs_max_val = jnp.pad(x_abs_max_val, ((0, 0), (0, padded_bs - orig_bs)))
  padded_out_features = _next_multiple(orig_out_features, out_block_size)
  if orig_out_features < padded_out_features:
    w = jnp.pad(w, ((0, padded_out_features - orig_out_features), (0, 0)))
    scalar = jnp.pad(scalar, (0, padded_out_features - orig_out_features))
  padded_in_features = _next_multiple(orig_in_features, in_block_size)
  if orig_in_features < padded_in_features:
    x = jnp.pad(x, ((0, 0), (0, padded_in_features - orig_in_features)))
    w = jnp.pad(w, ((0, 0), (0, padded_in_features - orig_in_features)))

  if scalar.dtype != jnp.float32:
    scalar = scalar.astype(jnp.float32)
  scalar = jnp.expand_dims(scalar, axis=0)  # [1, n_output_features]

  assert x.shape[1] == w.shape[
      1], f"x.shape[1] ({x.shape[1]}) must be equal to w.shape[1] ({w.shape[1]})"
  assert w.shape[0] == scalar.shape[
      1], f"w.shape[0] ({w.shape[0]}) must be equal to scalar.shape[1] ({scalar.shape[1]})"
  assert x_abs_max_val.shape == (
      1, x.shape[0]
  ), f"x_max_val.shape ({x_abs_max_val.shape}) must be equal to (1, x.shape[0]) ({1, {x.shape[0]}})"
  assert x.shape[
      0] % batch_block_size == 0, f"x.shape[0] ({x.shape[0]}) must be a multiple of block size ({batch_block_size})"
  assert w.shape[
      0] % out_block_size == 0, f"w.shape[0] ({w.shape[0]}) must be a multiple of block size ({out_block_size})"
  assert x.shape[
      1] % in_block_size == 0, f"x.shape[1] ({x.shape[1]}) must be a multiple of block size ({in_block_size})"

  acc_dtype = jnp.int32 if quantize_activation else x.dtype
  kernel = pl.pallas_call(
      functools.partial(
          matmul_kernel,
          quantize_activation=quantize_activation,
          batch_block_size=batch_block_size,
          out_block_size=out_block_size,
          in_block_size=in_block_size),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i:
                           (b, i)),
              pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i:
                           (o, i)),
              pl.BlockSpec((1, out_block_size), lambda b, o, i:
                           (0, o)),  # scalar
              pl.BlockSpec((1, batch_block_size), lambda b, o, i:
                           (0, b)),  # x_abs_max_val
          ],
          out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                 lambda b, o, i: (b, o)),
          scratch_shapes=[
              pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
          ],
          grid=(padded_bs // batch_block_size,
                padded_out_features // out_block_size,
                padded_in_features // in_block_size),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out_features), x.dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )

  out = kernel(x, w, scalar, x_abs_max_val)

  return out[:orig_bs, :orig_out_features]


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
    (6, 16, 6144, 4096, 'bfloat16', True): (128, 6144, 1024),
    (6, 16, 4096, 4096, 'bfloat16', True): (128, 4096, 1024),
    (6, 16, 28672, 4096, 'bfloat16', True): (128, 3584, 2048),
    (6, 16, 4096, 14336, 'bfloat16', True): (128, 4096, 1792),
    (6, 16, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 16, 8192, 1024, 'bfloat16', True): (128, 1024, 1024),
    (6, 16, 7168, 8192, 'bfloat16', True): (128, 7168, 1024),
    (6, 16, 8192, 3584, 'bfloat16', True): (128, 2048, 3584),
    (6, 32, 6144, 4096, 'bfloat16', True): (128, 1536, 4096),
    (6, 32, 4096, 4096, 'bfloat16', True): (128, 4096, 1024),
    (6, 32, 28672, 4096, 'bfloat16', True): (128, 4096, 2048),
    (6, 32, 4096, 14336, 'bfloat16', True): (128, 4096, 1792),
    (6, 32, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 32, 8192, 1024, 'bfloat16', True): (128, 8192, 256),
    (6, 32, 7168, 8192, 'bfloat16', True): (128, 1792, 4096),
    (6, 32, 8192, 3584, 'bfloat16', True): (128, 8192, 896),
    (6, 64, 6144, 4096, 'bfloat16', True): (128, 1536, 2048),
    (6, 64, 4096, 4096, 'bfloat16', True): (128, 2048, 1024),
    (6, 64, 28672, 4096, 'bfloat16', True): (128, 3584, 2048),
    (6, 64, 4096, 14336, 'bfloat16', True): (128, 4096, 1024),
    (6, 64, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 64, 8192, 1024, 'bfloat16', True): (128, 8192, 1024),
    (6, 64, 7168, 8192, 'bfloat16', True): (128, 3584, 2048),
    (6, 64, 8192, 3584, 'bfloat16', True): (128, 2048, 1792),
    (6, 128, 6144, 4096, 'bfloat16', True): (128, 6144, 1024),
    (6, 128, 4096, 4096, 'bfloat16', True): (128, 4096, 2048),
    (6, 128, 28672, 4096, 'bfloat16', True): (128, 28672, 512),
    (6, 128, 4096, 14336, 'bfloat16', True): (128, 2048, 3584),
    (6, 128, 1280, 8192, 'bfloat16', True): (128, 1280, 1024),
    (6, 128, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 128, 7168, 8192, 'bfloat16', True): (128, 1792, 4096),
    (6, 128, 8192, 3584, 'bfloat16', True): (128, 8192, 896),
    (6, 256, 6144, 4096, 'bfloat16', True): (256, 3072, 4096),
    (6, 256, 4096, 4096, 'bfloat16', True): (256, 2048, 4096),
    (6, 256, 28672, 4096, 'bfloat16', True): (256, 3584, 4096),
    (6, 256, 4096, 14336, 'bfloat16', True): (256, 4096, 1792),
    (6, 256, 1280, 8192, 'bfloat16', True): (256, 1280, 2048),
    (6, 256, 8192, 1024, 'bfloat16', True): (256, 4096, 1024),
    (6, 256, 7168, 8192, 'bfloat16', True): (256, 1792, 4096),
    (6, 256, 8192, 3584, 'bfloat16', True): (256, 8192, 512),
    (6, 512, 6144, 4096, 'bfloat16', True): (512, 2048, 4096),
    (6, 512, 4096, 4096, 'bfloat16', True): (512, 4096, 512),
    (6, 512, 28672, 4096, 'bfloat16', True): (512, 4096, 4096),
    (6, 512, 4096, 14336, 'bfloat16', True): (512, 4096, 2048),
    (6, 512, 1280, 8192, 'bfloat16', True): (512, 1280, 2048),
    (6, 512, 8192, 1024, 'bfloat16', True): (512, 4096, 1024),
    (6, 512, 7168, 8192, 'bfloat16', True): (512, 7168, 512),
    (6, 512, 8192, 3584, 'bfloat16', True): (512, 8192, 512),
    (6, 1024, 6144, 4096, 'bfloat16', True): (512, 6144, 4096),
    (6, 1024, 4096, 4096, 'bfloat16', True): (256, 4096, 4096),
    (6, 1024, 28672, 4096, 'bfloat16', True): (1024, 4096, 4096),
    (6, 1024, 4096, 14336, 'bfloat16', True): (1024, 4096, 1792),
    (6, 1024, 1280, 8192, 'bfloat16', True): (512, 1280, 4096),
    (6, 1024, 8192, 1024, 'bfloat16', True): (512, 8192, 1024),
    (6, 1024, 7168, 8192, 'bfloat16', True): (512, 7168, 1024),
    (6, 1024, 8192, 3584, 'bfloat16', True): (256, 8192, 3584),
    (6, 2048, 6144, 4096, 'bfloat16', True): (256, 6144, 4096),
    (6, 2048, 4096, 4096, 'bfloat16', True): (512, 4096, 4096),
    (6, 2048, 28672, 4096, 'bfloat16', True): (1024, 4096, 4096),
    (6, 2048, 4096, 14336, 'bfloat16', True): (1024, 4096, 2048),
    (6, 2048, 1280, 8192, 'bfloat16', True): (2048, 1280, 1024),
    (6, 2048, 8192, 1024, 'bfloat16', True): (256, 8192, 1024),
    (6, 2048, 7168, 8192, 'bfloat16', True): (256, 7168, 8192),
    (6, 2048, 8192, 3584, 'bfloat16', True): (512, 8192, 3584),
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
