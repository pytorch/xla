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

cases = {
  "TP=1":{
    "bs_nums": {16: 127, 128: 901, 256: 1, 512: 512, 1024: 5, 2048: 846},
    "out_in_features": [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)],
  },
  "TP=8": {
    "bs_nums": {16: 1016, 64: 64, 128: 1440, 512: 8, 1024: 32, 2048: 7032},
    "out_in_features": [(1280, 8192), (8192, 1024), (7168, 8192), (8192, 3584)],
  },
}

# one off, for testing
# cases = {
#   "TP=1":{
#     "bs_nums": {16: 1, 128: 2},
#     "out_in_features": [(128, 128)],
#   },
#   "TP=8": {
#     "bs_nums": {16: 1, 128: 2},
#     "out_in_features": [(256, 256)],
#   },
# }

def _quantize_tensor(x, n_bits: int = 8, dim: int = -1):
  max_val = jnp.amax(jnp.abs(x), axis=dim, keepdims=True)
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = max_val / int_max
  x_int = jnp.clip(jnp.rint(x / scale), int_min, int_max).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)

def run_benchmark(bs_nums, out_in_features: List[tuple]):
  elapsed_time_ms = 0
  print(f"Running benchmark with bs_nums: {bs_nums} and out_in_features: {out_in_features}")
  for bs, num_occur in bs_nums.items():
    for n_output_features, n_input_features in out_in_features:
      dtype = jnp.bfloat16
      prng_key = random.key(1234)
      k0, k1 = jax.random.split(prng_key, 2)
      x = jax.random.normal(k0, (bs, n_input_features), dtype=dtype)
      w = jax.random.normal(k1, (n_output_features, n_input_features), dtype=dtype)
      w_w8a8_jax, scalar_jax = _quantize_tensor(w, n_bits=8, dim=-1)
      scalar_jax = scalar_jax.squeeze()
      assert scalar_jax.shape == (n_output_features,)
      num_warmup = 5
      for _ in range(num_warmup):
        quantized_matmul_int8(x, w_w8a8_jax, scalar_jax, quantize_activation=True).block_until_ready()
      start_time = time.perf_counter_ns()
      for _ in range(num_occur):
        quantized_matmul_int8(x, w_w8a8_jax, scalar_jax, quantize_activation=True).block_until_ready()
      end_time = time.perf_counter_ns()
      elapsed_time_ms += (end_time - start_time)/(1e6)
  return elapsed_time_ms

for case, value in cases.items():
  bs_nums = value["bs_nums"]
  out_in_features = value["out_in_features"]
  elapsed_time_ms = run_benchmark(bs_nums, out_in_features)
  print(f"Benchmarking {case} took {elapsed_time_ms:.2f} ms")
