import functools
import timeit
import numpy as np
import jax
from jax import numpy as jnp
from jax import lax
from jax.experimental import checkify
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

assert "TPU" in jax.devices()[0].device_kind, "Please run this notebook with TPU devices."
print("Running on", jax.devices()[0].device_kind)



def ceil_div(a, b):
  assert b != 0
  return (a + b - 1) // b


def kv_insertion_kernel(
  # Prefetch
  slices_ref, # [num_slices, 2]
  cu_kv_lens_ref, # [num_slices + 1]
  slice_idx_ref, # [1]
  # Input
  k_ref, # [num_token_per_blk, kv_hidden_size]
  _,
  # Output
  k_pages_hbm_ref, # [total_num_pages * page_size, kv_hidden_size]
):
  blk_idx = pl.program_id(0)
  num_token_per_blk = k_ref.shape[0]
  num_slices = slices_ref.shape[0]
  init_slice_idx = slice_idx_ref[0]
  kv_block_start = blk_idx * num_token_per_blk
  kv_block_end = kv_block_start + num_token_per_blk

  def is_cur_slice_needed(slice_states):
    done, cur_slice_idx = slice_states
    return jnp.logical_and(done == 0, cur_slice_idx < num_slices)

  def insert_cur_k_blk(slice_states):
    done, cur_slice_idx = slice_states
    kv_start = cu_kv_lens_ref[cur_slice_idx]
    kv_end = cu_kv_lens_ref[cur_slice_idx + 1]

    def masked_load(ref, start, end):
      iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0)
      mask = jnp.logical_and(iota >= start, iota < end)
      return pl.load(ref, tuple(slice(None) for _ in ref.shape), mask=mask)

    load_start = jnp.maximum(kv_start - kv_block_start, 0)
    load_end = jnp.minimum(kv_end - kv_block_start, num_token_per_blk)
    cache_value = masked_load(k_ref, load_start, load_end)
    kv_store_start, kv_store_size = slices_ref[cur_slice_idx]
    pl.store(k_pages_hbm_ref, 
             (pl.ds(kv_store_start, kv_store_size), slice(None)),
             cache_value
    )

    next_slice_idx = cur_slice_idx + 1
    done = lax.select(kv_end < kv_block_end, done, 1)
    return done, next_slice_idx
  
  _, next_slice_idx = lax.while_loop(
    is_cur_slice_needed,
    insert_cur_k_blk,
    (0, init_slice_idx)
  )

  slice_idx_ref[0] = next_slice_idx

# kv_hidden_size = kv_head_num * head_size
@functools.partial(
  jax.jit,
  static_argnames=[
    "num_token_per_block",
  ],
)
def kv_insertion(
  k: jax.Array, # [total_num_token, kv_hidden_size]
  slices: jax.Array, # [num_slices, 2], list of (start, size)
  k_pages: jax.Array, # [total_num_pages * page_size, kv_hidden_size]
  *,
  num_token_per_block = 16,
):
  cu_slices_lens = slices[1] - slices[0]
  cu_slices_lens = jnp.cumsum(cu_slices_lens)
  cu_slices_lens = jnp.concatenate((jnp.array([0]), cu_slices_lens))

  total_num_token, kv_hidden_size = k.shape
  num_k_blks = ceil_div(total_num_token, num_token_per_block)
  grid = (num_k_blks, )

  def k_index_map(blk_idx, *_):
    return (blk_idx, 0)
  k_block_spec = pl.BlockSpec(
    (num_token_per_block, kv_hidden_size),
    k_index_map,
  )

  in_specs = [
    k_block_spec,
    pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
  ]

  out_specs = [pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)]
  out_shape = [jax.ShapeDtypeStruct(k_pages.shape, dtype=k_pages.dtype)]

  scalar_prefetches = [slices, cu_slices_lens, jnp.array((0, ), jnp.int32)]

  kernel = pl.pallas_call(
    kv_insertion_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=len(scalar_prefetches),
      in_specs=in_specs,
      out_specs=out_specs,
      grid=grid,
    ),
    out_shape=out_shape,
    input_output_aliases={4: 0},
  )
  
  return kernel(*scalar_prefetches, k, k_pages)[0]


if __name__ == "__main__":
  total_num_pages = 10000
  page_size = 16
  kv_hidden_size = 1024
  k_pages = jnp.zeros((total_num_pages * page_size, kv_hidden_size), dtype=jnp.bfloat16)
  k = jnp.ones((1024, kv_hidden_size), dtype=jnp.bfloat16)
  slices = jnp.array([[0, 1024]])
  out = kv_insertion(k, slices, k_pages)
  print(out)
