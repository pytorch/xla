import functools
from typing import cast
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

def _quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_max_val: jax.Array,  # [1, bs_block_size]
    n_bits: int = 8,
):
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = (x_max_val / int_max).T  # [bs_block_size, 1]
  # Need to explicitly cast to f32 because Mosaic can't directly jnp.round a
  # bf16 array.
  # It seems x/0 in Pallas generates inf/-inf instead of an exception.
  x_int = jnp.clip(jnp.round((x / scale).astype(jnp.float32)), int_min, int_max).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)

def matmul_kernel(
    x_ref,  # (batch_block_size, in_block_size)
    w_ref,  # (out_block_size, in_block_size)
    scalar_ref,  # (1, out_block_size)
    x_max_val,  # (1, batch_block_size)
    out_ref,  # (batch_block_size, out_block_size)
    acc_ref,  # (batch_block_size, out_block_size)
    *,
    quantize_activation,
    batch_block_size,
    out_block_size,
    in_block_size,
  ):
  bs_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1) , pl.program_id(2)
  nsteps = pl.num_programs(2)
  x_ref_dtype = x_ref.dtype
  assert x_ref.shape == (batch_block_size, in_block_size), "x_ref shape is not correct"
  assert w_ref.shape == (out_block_size, in_block_size), "w_ref shape is not correct"
  assert scalar_ref.shape == (1, out_block_size), "scalar_ref shape is not correct"
  assert x_max_val.shape == (1, batch_block_size), "x_max_val shape is not correct"
  assert out_ref.shape == (batch_block_size, out_block_size), "out_ref shape is not correct"
  assert acc_ref.shape == (batch_block_size, out_block_size), "acc_ref shape is not correct"

  @pl.when(in_idx == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  if quantize_activation:
    x, x_scale = _quantize_array(x_ref[...], x_max_val[...])
    acc_ref[...] += jax.lax.dot_general(
        x,
        w_ref[...],
        (((1,), (1,)),((), ())),
        preferred_element_type=jnp.int32,
    )
  else:
    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        (((1,), (1,)),((), ())),
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
        'int4_weight',
        'quantize_activation',
        'batch_block_size',
        'out_block_size',
        'in_block_size',
    ]
)
def quantized_matmul(
    x: jax.Array,  # [bs, n_input_features]
    w: jax.Array,  # [n_output_features, n_input_features]
    scalar: jax.Array,  # [n_output_features]
    zero_point: jax.Array | None = None,
    block_size: jax.Array | None = None,  # i32[1]
    int4_weight: bool = False,
    quantize_activation: bool = False,
    *,
    # All 3 block sizes have to be multiples of 128 because they are used as the minormost dimension in the block.
    batch_block_size: int = 128,
    out_block_size: int = 128,
    in_block_size: int = 128,
    vmem_limit_bytes: int | None = 64 * 1024 * 1024,
):
  assert zero_point is None, "Not implemented: zero_point is not supported."
  assert block_size is None, "Not implemented: block_size is not supported."
  assert not int4_weight, "Not implemented: int4_weight is not supported."

  # x_max_val cannot be [bs, 128] because it'll be costly to send
  # [bs_block_size, 128] to VMEM each time.
  # cast to f32 because "INTERNAL: Mosaic failed to compile TPU kernel: Insertion of minor dim that is not a no-op only supported for 32-bit types"
  # We need the global max values to be computed before the kernel.
  x_max_val = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
  x_max_val = jnp.expand_dims(x_max_val, axis=0)  # [1, bs]
  assert x_max_val.shape == (1, x.shape[0])

  orig_bs, orig_in_features = x.shape
  orig_out_features, _ = w.shape
  padded_bs = _next_multiple(orig_bs, batch_block_size)
  if orig_bs < padded_bs:
    x = jnp.pad(x, ((0, padded_bs-orig_bs), (0, 0)))
    x_max_val = jnp.pad(x_max_val, ((0, 0), (0, padded_bs-orig_bs)))
  padded_out_features = _next_multiple(orig_out_features, out_block_size)
  if orig_out_features <  padded_out_features:
    w = jnp.pad(w, ((0, padded_out_features-orig_out_features), (0, 0)))
    scalar = jnp.pad(scalar, (0, padded_out_features-orig_out_features))
  padded_in_features = _next_multiple(orig_in_features, in_block_size)
  if orig_in_features < padded_in_features:
    x = jnp.pad(x, ((0, 0), (0, padded_in_features-orig_in_features)))
    w = jnp.pad(w, ((0, 0), (0, padded_in_features-orig_in_features)))

  scalar = jnp.expand_dims(scalar, axis=0)  # [1, n_output_features]

  assert x.shape[1] == w.shape[1], f"x.shape[1] ({x.shape[1]}) must be equal to w.shape[1] ({w.shape[1]})"
  assert w.shape[0] == scalar.shape[1], f"w.shape[0] ({w.shape[0]}) must be equal to scalar.shape[1] ({scalar.shape[1]})"
  assert x_max_val.shape == (1, x.shape[0]), f"x_max_val.shape ({x_max_val.shape}) must be equal to (1, x.shape[0]) ({1, {x.shape[0]}})"
  assert x.shape[0] % batch_block_size == 0, f"x.shape[0] ({x.shape[0]}) must be a multiple of block size ({batch_block_size})"
  assert w.shape[0] % out_block_size == 0, f"w.shape[0] ({w.shape[0]}) must be a multiple of block size ({out_block_size})"
  assert x.shape[1] % in_block_size == 0, f"x.shape[1] ({x.shape[1]}) must be a multiple of block size ({in_block_size})"

  acc_dtype = jnp.int32 if quantize_activation else x.dtype
  kernel = pl.pallas_call(
      functools.partial(matmul_kernel, quantize_activation=quantize_activation, batch_block_size=batch_block_size, out_block_size=out_block_size, in_block_size=in_block_size),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i: (b, i)),
              pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i: (o, i)),
              pl.BlockSpec((1, out_block_size), lambda b, o, i: (0, o)),
              pl.BlockSpec((1, batch_block_size), lambda b, o, i: (0, b)),
          ],
          out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
          scratch_shapes=[pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)],
          grid=(padded_bs // batch_block_size, padded_out_features // out_block_size, padded_in_features // in_block_size),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out_features), x.dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )

  out = kernel(x, w, scalar, x_max_val)

  return out[:orig_bs,:orig_out_features]

def quantize_array(x, n_bits: int = 8, dim: int = -1):
  max_val = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = max_val / int_max
  x_int = jnp.clip(jnp.round((x / scale)), int_min, int_max).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)
