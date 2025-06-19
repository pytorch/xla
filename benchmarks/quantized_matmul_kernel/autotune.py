# To run, do python myscripts/autotune_quantized_matmul_pallas_kernel2.py 2>&1 | tee out.txt
# Then in the out.txt, extract lines with "Add to table:" and replace the string with 4 spaces, then copy to the block table in the pallas kernel.
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
    get_tuned_block_sizes,
    TUNED_BLOCK_SIZES,
)

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

    vmem_limit_80_mb = 80 * 1024 * 1024
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
          for _ in range(10):  # warming up
            try:
              quantized_matmul_int8(x, w_w8a8_jax, scalar_jax, quantize_activation=True, batch_block_size=batch_block_size, out_block_size=out_block_size, in_block_size=in_block_size, vmem_limit_bytes=vmem_limit_80_mb).block_until_ready()
            except Exception as e:
              print(f'Failed to run quantized_matmul with batch_block_size={batch_block_size}, out_block_size={out_block_size}, in_block_size={in_block_size} due to {e}', flush=True)
              skip_trial = True
              break
          if skip_trial:
            continue
          num_iterations = 30
          start_time = time.perf_counter_ns()
          for _ in range(num_iterations):
            quantized_matmul_int8(x, w_w8a8_jax, scalar_jax, quantize_activation=True, batch_block_size=batch_block_size, out_block_size=out_block_size, in_block_size=in_block_size, vmem_limit_bytes=vmem_limit_80_mb).block_until_ready()
          end_time = time.perf_counter_ns()
          elapsed_time = (end_time - start_time) / num_iterations
          print(f'Benchmarked w8a8 matmul with batch_block_size={batch_block_size}, out_block_size={out_block_size}, in_block_size={in_block_size}, time={elapsed_time}')
          if best_time is None or elapsed_time < best_time:
            best_time = elapsed_time
            best_batch_block_size = batch_block_size
            best_out_block_size = out_block_size
            best_in_block_size = in_block_size
    print(f'Best batch_block_size={best_batch_block_size}, out_block_size={best_out_block_size}, in_block_size={best_in_block_size}, time={best_time}')
    print(f'Add to table: (6, {bs}, {n_output_features}, {n_input_features}, \'{jnp.dtype(dtype).name}\', {True}): ({best_batch_block_size}, {best_out_block_size}, {best_in_block_size}),')

# key should be: bs, n_output_features, n_input_features, dtype, quantize_activation
