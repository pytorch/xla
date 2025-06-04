import time
from typing import List

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import numpy as np
import functools
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_xla.experimental.pallas_kernels.quantized_matmul_kernel import (
    quantized_matmul_int8,
)


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
    acc *= scalar_ref[...]
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
  # compiled_kernel = (
  #       jax.jit(kernel)
  #       .lower(x, w, scalar, x_max_val)
  #       .compile({'xla_tpu_enable_log_recorder': 'true'})
  # )
  # out = compiled_kernel(x, w, scalar, x_max_val)

  return out[:orig_bs,:orig_out_features]

def quantize_array(x, n_bits: int = 8, dim: int = -1):
  max_val = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = max_val / int_max
  # print('xw32 debug ref _quantize_array: x={}', x)
  # print('xw32 debug ref _quantize_array: scale={}', scale)
  # print('xw32 debug ref _quantize_array: int_min={}, int_max={}', int_min, int_max)
  x_int = jnp.clip(jnp.round((x / scale)), int_min, int_max).astype(jnp.int8)
  # x_int = jnp.clip(jnp.rint((x.astype(jnp.float32) / scale.astype(jnp.float32))), int_min, int_max).astype(jnp.bfloat16).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)

def _quantize_tensor(x, n_bits: int = 8, dim: int = -1):
  max_val = jnp.amax(jnp.abs(x), axis=dim, keepdims=True)
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = max_val / int_max
  x_int = jnp.clip(jnp.rint(x / scale), int_min, int_max).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)

def find_factors_multiple_of_128(n: int) -> List[int]:
    """
    Finds all factors of an integer n that are also multiples of 128.

    Args:
        n: The integer for which to find factors.

    Returns:
        A list of integers that are factors of n and are multiples of 128.
        Returns an empty list if n is 0 or if no such factors exist.
        Handles negative input by taking the absolute value.
    """
    # Handle edge case for 0
    if n == 0:
        return []

    # Work with the absolute value for factor finding
    n_abs = abs(n)

    # We are looking for factors f such that n_abs % f == 0 AND f % 128 == 0.
    # This means f must be a multiple of 128.
    # So, we only need to check multiples of 128 up to n_abs.

    factors = []
    multiplier = 1
    potential_factor = 128 * multiplier

    # Iterate through multiples of 128 (128, 256, 384, ...)
    # as long as they are less than or equal to n_abs
    while potential_factor <= n_abs:
        # Check if this multiple of 128 is a factor of n_abs
        if n_abs % potential_factor == 0:
            factors.append(potential_factor)

        # Move to the next multiple of 128
        multiplier += 1
        potential_factor = 128 * multiplier # Calculate the next potential factor

    return factors

# Benchmarking script starts.

# one off
# batch_sizes = [16, 32]
# out_in_features = [(128, 256), (256, 128)]
# batch_block_sizes = [128, 256]

# for real
batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
out_in_features = [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336), (1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584)]
batch_block_sizes = [128, 256, 512, 1024, 2048]

for bs in batch_sizes:
  for n_output_features, n_input_features in out_in_features:
    out_block_sizes = find_factors_multiple_of_128(n_output_features)
    in_block_sizes = find_factors_multiple_of_128(n_input_features)
    print(f'Benchmarking w8a8 matmul with bs={bs}, n_output_features={n_output_features}, n_input_features={n_input_features}, for block sizes: {out_block_sizes=}, {in_block_sizes=}')

    dtype = jnp.bfloat16
    prng_key = jax.random.key(1234)
    k0, k1 = jax.random.split(prng_key, 2)
    x = jax.random.normal(k0, (bs, n_input_features), dtype=dtype)
    w = jax.random.normal(k1, (n_output_features, n_input_features), dtype=dtype)
    w_w8a8_jax, scalar_jax = _quantize_tensor(w, n_bits=8, dim=-1)
    scalar_jax = scalar_jax.squeeze()
    assert scalar_jax.shape == (n_output_features,)

    best_time = None
    best_batch_block_size = None
    best_out_block_size = None
    best_in_block_size = None
    skip_trial = False
    for batch_block_size in batch_block_sizes:
      if bs < batch_block_size and best_time is not None:
        continue
      for out_block_size in out_block_sizes:
        for in_block_size in in_block_sizes:
          skip_trial = False
          print(f'Benchmarking w8a8 matmul bs={bs}, n_output_features={n_output_features}, n_input_features={n_input_features} with batch_block_size={batch_block_size}, out_block_size={out_block_size}, in_block_size={in_block_size}', flush=True)
          for _ in range(5):  # warming up
            try:
              quantized_matmul_int8(x, w_w8a8_jax, scalar_jax, quantize_activation=True, batch_block_size=batch_block_size, out_block_size=out_block_size, in_block_size=in_block_size).block_until_ready()
            except Exception as e:
              print(f'Failed to run quantized_matmul with batch_block_size={batch_block_size}, out_block_size={out_block_size}, in_block_size={in_block_size} due to {e}', flush=True)
              skip_trial = True
              break
          if skip_trial:
            continue
          num_iterations = 20
          start_time = time.perf_counter_ns()
          for _ in range(num_iterations):
            quantized_matmul_int8(x, w_w8a8_jax, scalar_jax, quantize_activation=True, batch_block_size=batch_block_size, out_block_size=out_block_size, in_block_size=in_block_size).block_until_ready()
          end_time = time.perf_counter_ns()
          elapsed_time = (end_time - start_time) / num_iterations
          print(f'Benchmarked w8a8 matmul with batch_block_size={batch_block_size}, out_block_size={out_block_size}, in_block_size={in_block_size}, time={elapsed_time}')
          if best_time is None or elapsed_time < best_time:
            best_time = elapsed_time
            best_batch_block_size = batch_block_size
            best_out_block_size = out_block_size
            best_in_block_size = in_block_size
    print(f'Best batch_block_size={best_batch_block_size}, out_block_size={best_out_block_size}, in_block_size={best_in_block_size}, time={best_time}')
    print(f'Add to table: ({bs}, {n_output_features}, {n_input_features}, {jnp.dtype(dtype).name}, {True}): ({best_batch_block_size}, {best_out_block_size}, {best_in_block_size})')

# key should be: bs, n_output_features, n_input_features, dtype, quantize_activation
