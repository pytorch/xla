import argparse
import time

import jax
from jax._src import test_util as jtu
from torch_xla.experimental.pallas_kernels.multi_queries_paged_attention_kernel import paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention as jax_single_query_paged_attention
import jax.numpy as jnp
import numpy as np

# Set up paged_attention inputs.
def _generate_qkv(
    kv_seq_lens,
    page_size,
    max_kv_len,
    query_len,
    num_kv_heads,
    num_q_heads,
    head_dim,
    prng_key,
    dtype,
    total_pages,
):
  # assert max_kv_len % page_size == 0
  pages_per_sequence = (max_kv_len+page_size-1) // page_size
  batch_size = len(kv_seq_lens)
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(
      k4, (batch_size, query_len, num_q_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices

def benchmark(args):
  # Shapes and dtypes used in the GPU benchmarking script: query.shape=torch.Size([36, 8, 256]), key_cache.shape=torch.Size([231746, 16, 1, 256]), value_cache.shape=torch.Size([231746, 16, 1, 256]), prefill_meta.query_start_loc=tensor([ 0,  9, 18, 27, 36], device='cuda:0', dtype=torch.int32), prefill_meta.max_query_len=9, prefill_meta.seq_start_loc=tensor([   0,  649, 1298, 1947, 2596], device='cuda:0', dtype=torch.int32), max_seq_len=649, softmax_scale=0.0625, window_size=[-1, -1], alibi_slopes=None, prefill_meta.block_tables.shape=torch.Size([4, 41]), logits_soft_cap=0.0
  assert args.kernel == "multi-queries-paged-attn" or args.kernel == "single-query-flash-attn", f"invalid args.kernel: {args.kernel}"
  dtype = jnp.bfloat16
  page_size = 16
  num_kv_heads = 2
  q_kv_head_ratio = 4
  head_dim = 128
  block_kv_size = 256

  kv_seq_lens_lst = [1328, 18, 463]
  q_seq_lens_lst = [1, 5, 129]
  kv_seq_lens = jnp.array(kv_seq_lens_lst)
  effective_q_lens = jnp.array(q_seq_lens_lst)
  max_kv_len = 2048 # max(kv_seq_lens_lst), the change is needed to make pages_per_sequence a multiple of num_kv_pages_per_compute_block. TODO: adjust in CUDA
  query_len = max(q_seq_lens_lst)
  total_num_pages = 2048  # in vLLM CUDA flash_attention benchmarking script, this is tunable parameter.
  assert max_kv_len <= total_num_pages * page_size

  q, k_pages, v_pages, page_indices = _generate_qkv(
      kv_seq_lens,
      page_size,
      max_kv_len,
      query_len if args.kernel == "multi-queries-paged-attn" else 1,
      num_kv_heads,
      num_kv_heads * q_kv_head_ratio,
      head_dim,
      jax.random.key(0),
      dtype,
      total_num_pages,
  )

  def run_benchmark(num_iters: int, profile: bool = False) -> float:
    start_time = time.perf_counter()

    actual_output=None
    for _ in range(num_iters):
      if args.kernel == "multi-queries-paged-attn":
        num_queries_per_compute_block=16 # constraint: due to https://github.com/jax-ml/jax/issues/24486
        num_kv_pages_per_compute_block = block_kv_size // page_size
        actual_output = paged_attention(
          q,
          k_pages,
          v_pages,
          kv_seq_lens,
          page_indices,
          effective_q_lens,
          num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
          num_queries_per_compute_block=num_queries_per_compute_block,
        )
      elif args.kernel == "single-query-flash-attn":
        actual_output = jax_single_query_paged_attention(
          jnp.squeeze(q, axis=1),
          k_pages,
          v_pages,
          kv_seq_lens,
          page_indices,
          pages_per_compute_block=16,
        )

    jax.block_until_ready(actual_output)
    end_time = time.perf_counter()
    return (end_time - start_time) / num_iters
  
  # Warmup.
  print("Warming up...")
  run_benchmark(num_iters=3, profile=False)

  print("Run benchmark...")
  if args.profile:
    latency = run_benchmark(num_iters=1, profile=True)
  else:
    latency = run_benchmark(num_iters=10, profile=False)
  print(f"Kernel running time: {latency * 1000000:.3f} us")
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--use-paged-attn-nonkernel", action="store_true")
  parser.add_argument("--kernel",
                type=str,
                choices=["single-query-paged-attn", "multi-queries-paged-attn"],
                default="multi-queries-paged-attn")
  parser.add_argument("--profile", action="store_true")
  args = parser.parse_args()
  benchmark(args)
