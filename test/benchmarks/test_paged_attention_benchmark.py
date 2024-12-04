# python pytorch/xla/test/benchmarks/test_paged_attention_benchmark.py --kernel multi-queries-paged-attn-jax-nonkernel
# python pytorch/xla/test/benchmarks/test_paged_attention_benchmark.py --kernel multi-queries-paged-attn

import argparse
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.custom_kernel import multi_queries_paged_attention
import jax
from jax._src import test_util as jtu
from torch_xla.experimental.pallas_kernels.multi_queries_paged_attention_kernel import paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention as jax_single_query_paged_attention
import jax.numpy as jnp
import numpy as np

@jax.profiler.annotate_function
def _ref_jax_extended_paged_attention(
    q,  # [batch_size, query_len, num_query_heads, head_size]
    k_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    v_pages,  # [num_kv_heads, total_num_pages, page_size, head_size]
    lengths,  # [batch_size], the effective kv_length.
    page_indices,  # [batch_size, pages_per_sequence]
    effective_q_lens,  # [batch_size] the effective q_length
):
  batch_size, query_len, num_query_heads, head_size = q.shape
  num_kv_heads, total_num_pages, page_size, _ = k_pages.shape
  num_query_per_kv = num_query_heads // num_kv_heads

  outputs = []
  for i in range(batch_size):
    kv_len = lengths[i]
    num_pages = (kv_len + page_size - 1) // page_size
    indices = page_indices[i, :num_pages]

    k = k_pages[:, indices]
    k = jnp.permute_dims(k, (1, 2, 0, 3))
    k = jnp.reshape(k, (num_pages * page_size, num_kv_heads, head_size))
    k = k[:kv_len]

    v = v_pages[:, indices]
    v = jnp.permute_dims(v, (1, 2, 0, 3))
    v = jnp.reshape(v, (num_pages * page_size, num_kv_heads, head_size))
    v = v[:kv_len]

    if num_query_per_kv != 1:
      k = jnp.repeat(k, num_query_per_kv, axis=1)
      v = jnp.repeat(v, num_query_per_kv, axis=1)

    attn = jnp.einsum("qhd,khd->hqk", q[i], k)
    attn = attn.astype('float32')
    effective_q_len = effective_q_lens[i]
    q_span = (kv_len - effective_q_len) + jax.lax.broadcasted_iota(
        jnp.int32, (query_len, kv_len), 0)
    kv_span = jax.lax.broadcasted_iota(jnp.int32, (query_len, kv_len), 1)
    mask = jnp.where(q_span < kv_span, float("-inf"), 0.)
    with jax.numpy_rank_promotion("allow"):
      attn = attn + mask
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v)
    outputs.append(out)
  output = jnp.stack(outputs, axis=0)
  return output


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
  # Shapes and dtypes used in the GPU benchmarking script: xw32 line890, query.shape=torch.Size([36, 8, 256]), query.dtype=torch.bfloat16, key_cache.shape=torch.Size([231746, 16, 1, 256]), value_cache.shape=torch.Size([231746, 16, 1, 256]), prefill_meta.query_start_loc=tensor([ 0,  9, 18, 27, 36], device='cuda:0', dtype=torch.int32), prefill_meta.max_query_len=9, prefill_meta.seq_start_loc=tensor([   0,  649, 1298, 1947, 2596], device='cuda:0', dtype=torch.int32), max_seq_len=649, softmax_scale=0.0625, window_size=[-1, -1], alibi_slopes=None, prefill_meta.block_tables.shape=torch.Size([4, 41]), logits_soft_cap=0.0, query.dtype=torch.bfloat16
  dtype = jnp.bfloat16
  page_size = 16
  num_kv_heads = 1
  q_kv_head_ratio = 4
  head_dim = 256
  block_kv_size = 256

  kv_seq_lens_lst = [649, 649, 649, 649]
  q_seq_lens_lst = [16, 16, 16, 16] # num_queries_per_compute_block=16 should be smaller or equal to query_len=9
  kv_seq_lens = jnp.array(kv_seq_lens_lst)
  effective_q_lens = jnp.array(q_seq_lens_lst)
  # num_kv_pages_per_compute_block = block_kv_size // page_size
  # pages_per_sequence = max_kv_len // page_size
  max_kv_len = 768 # max(kv_seq_lens_lst), the change is needed to make pages_per_sequence a multiple of num_kv_pages_per_compute_block. TODO: adjust in CUDA
  query_len = max(q_seq_lens_lst)
  total_num_pages = 231746  # in vLLM CUDA flash_attention benchmarking script, this is tunable parameter.
  assert max_kv_len <= total_num_pages * page_size

  q, k_pages, v_pages, page_indices = _generate_qkv(
      kv_seq_lens,
      page_size,
      max_kv_len,
      query_len if args.kernel == "multi-queries-paged-attn" or args.kernel.startswith("multi-queries-paged-attn-torch-xla") else 1,
      num_kv_heads,
      num_kv_heads * q_kv_head_ratio,
      head_dim,
      jax.random.key(0),
      dtype,
      total_num_pages,
  )

  # import pdb; pdb.set_trace()
  # q_xla = torch.from_numpy(np.array(q)).to("xla")
  # k_pages_xla = torch.from_numpy(np.array(k_pages)).to("xla")
  # v_pages_xla = torch.from_numpy(np.array(v_pages)).to("xla")
  # kv_seq_lens_xla = torch.from_numpy(np.array(kv_seq_lens)).to("xla")
  # page_indices_xla = torch.from_numpy(np.array(page_indices)).to("xla")
  # effective_q_lens_xla = torch.from_numpy(np.array(effective_q_lens)).to("xla")
  profile_path = "/workspaces/persist/myprofiles/plugins/profile"

  def run_benchmark(num_iters: int, profile: bool = False) -> float:
    start_time = time.perf_counter()
    if profile:
      jax.profiler.start_trace(profile_path)

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
      if args.kernel == "multi-queries-paged-attn-jax-nonkernel":
        actual_output = _ref_jax_extended_paged_attention(
          q,
          k_pages,
          v_pages,
          kv_seq_lens,
          page_indices,
          effective_q_lens,
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
      # elif args.kernel == "multi-queries-paged-attn-torch-xla-kernel":
      #   num_queries_per_compute_block=16 # constraint: due to https://github.com/jax-ml/jax/issues/24486
      #   num_kv_pages_per_compute_block = block_kv_size // page_size
      #   actual_output = multi_queries_paged_attention(
      #       q_xla,
      #       k_pages_xla,
      #       v_pages_xla,
      #       kv_seq_lens_xla,
      #       page_indices_xla,
      #       effective_q_lens_xla,
      #       num_kv_pages_per_compute_block=block_kv_size // page_size,
      #       num_queries_per_compute_block=num_queries_per_compute_block,
      #       use_kernel=True,
      #   )
      # elif args.kernel == "multi-queries-paged-attn-torch-xla-nonkernel":
      #   num_queries_per_compute_block=16 # constraint: due to https://github.com/jax-ml/jax/issues/24486
      #   num_kv_pages_per_compute_block = block_kv_size // page_size
      #   actual_output = multi_queries_paged_attention(
      #       q_xla,
      #       k_pages_xla,
      #       v_pages_xla,
      #       kv_seq_lens_xla,
      #       page_indices_xla,
      #       effective_q_lens_xla,
      #       num_kv_pages_per_compute_block=block_kv_size // page_size,
      #       num_queries_per_compute_block=num_queries_per_compute_block,
      #       use_kernel=False,
      #   )


    # if args.kernel != "multi-queries-paged-attn-torch-xla-kernel" or args.kernel != "multi-queries-paged-attn-torch-xla-nonkernel":
    #   jax.block_until_ready(actual_output)
    # else:
    #   xm.mark_step()
    #   xm.wait_device_ops()
    jax.block_until_ready(actual_output)

    end_time = time.perf_counter()
    if profile:
      jax.profiler.stop_trace()
    return (end_time - start_time) / num_iters
  
  # Warmup.
  print("Warming up...")
  run_benchmark(num_iters=3, profile=False)

  print("Run benchmark...")
  if args.profile:
    latency = run_benchmark(num_iters=1, profile=True)
  else:
    latency = run_benchmark(num_iters=20, profile=False)
  print(f"Kernel running time: {latency * 1000000:.3f} us")
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--use-paged-attn-nonkernel", action="store_true")
  parser.add_argument("--kernel",
                type=str,
                choices=["single-query-paged-attn", "multi-queries-paged-attn","multi-queries-paged-attn-jax-nonkernel", "multi-queries-paged-attn-torch-xla-kernel", "multi-queries-paged-attn-torch-xla-nonkernel"],
                default="multi-queries-paged-attn")
  parser.add_argument("--profile", action="store_true")
  args = parser.parse_args()
  benchmark(args)
