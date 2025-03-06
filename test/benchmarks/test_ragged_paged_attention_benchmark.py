# Usage: python pytorch/xla/test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged-paged-attention-with-torch-xla-dynamo
# python pytorch/xla/test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged-paged-attention-with-torch-xla-nondynamo
# python pytorch/xla/test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged-paged-attention

import argparse
import time
from typing import List, Optional, Tuple
import functools
import os
import sys

import torch
import torch_xla.debug.profiler as xp
import torch_xla.experimental.custom_kernel  # Required to register custom ops.
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import numpy as np

if xr.device_type() == 'TPU':
  from torch_xla.experimental.custom_kernel import jax_import_guard
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl
  from torch_xla.experimental.pallas_kernels.ragged_paged_attention_kernel import ragged_paged_attention, make_sequence_metadata, DEFAULT_MASK_VALUE


def _ref_ragged_paged_attention(
    queries: jax.Array,  # [num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    v_pages: jax.Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    kv_lens: jax.Array,  # i32[num_tokens]
    page_indices: jax.Array,  # i32[num_tokens, pages_per_sequence]
    cu_q_lens: jax.Array,  # i32[num_tokens + 1]
    num_seqs: int,
):
  """This is the reference ragged paged attention implementation."""
  num_kv_heads, _, page_size, head_dim = k_pages.shape
  num_q_heads = queries.shape[1]
  assert num_q_heads % num_kv_heads == 0, "num_q_heads % num_kv_heads !=0."
  num_query_per_kv = num_q_heads // num_kv_heads
  start_idx = 0
  outputs: List[jax.Array] = []
  for i in range(num_seqs):
    cur_q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    q = queries[start_idx:start_idx +
                cur_q_len]  # [cur_q_len, num_q_heads, head_dim]

    cur_kv_len = kv_lens[i]
    num_pages = (cur_kv_len + page_size - 1) // page_size
    page_indices_to_use = page_indices[i, :num_pages]
    k = k_pages[:,
                page_indices_to_use, :, :]  # [num_kv_heads, page_indices_to_use, page_size, head_dim]
    k = jnp.permute_dims(
        k, (1, 2, 0,
            3))  # [page_indices_to_use, page_size, num_kv_heads, head_dim]
    k = jnp.reshape(
        k, (-1, num_kv_heads, head_dim))  #   [kv_len, num_kv_heads, head_dim]
    k = k[:cur_kv_len]  # [cur_kv_lens, num_kv_heads, head_dim]

    v = v_pages[:, page_indices_to_use, :, :]
    v = jnp.permute_dims(v, (1, 2, 0, 3))
    v = jnp.reshape(v, (-1, num_kv_heads, head_dim))
    v = v[:cur_kv_len]  # [cur_kv_lens, num_kv_heads, head_dim]

    if num_query_per_kv != 1:
      k = jnp.repeat(k, num_query_per_kv, axis=1)
      v = jnp.repeat(v, num_query_per_kv, axis=1)

    attn = jnp.einsum("qhd,khd->hqk", q, k)
    attn = attn.astype('float32')
    q_span = (cur_kv_len - cur_q_len) + jax.lax.broadcasted_iota(
        jnp.int32, (cur_q_len, cur_kv_len), 0)
    kv_span = jax.lax.broadcasted_iota(jnp.int32, (cur_q_len, cur_kv_len), 1)
    # Use the same DEFAULT_MASK_VALUE as in the kernel instead of float("-inf") so that the kernel can match the ref implement better.
    mask = jnp.where(q_span < kv_span, DEFAULT_MASK_VALUE, 0.)
    with jax.numpy_rank_promotion("allow"):
      attn = attn + mask
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn,
                     v)  # [cur_q_len, num_q_heads, head_dim]

    outputs.append(out)
    start_idx += cur_q_len

  return jnp.concatenate(outputs, axis=0)


def _get_closest_of_multiple(x, base):
  return (x + base - 1) // base * base


def _run_with_torch_xla(kernel):
  return "torch-xla" in kernel

def benchmark_decode(args):
  # xw32 pallas not allclose query.shape=torch.Size([16, 32, 128]),key_cache.shape=torch.Size([6112, 16, 8, 128]),value_cache.shape=torch.Size([6112, 16, 8, 128]),attn_metadata.context_lens.shape=torch.Size([1024]),attn_metadata.block_tables.shape=torch.Size([1024, 128]),attn_metadata.query_start_loc.shape=torch.Size([1025]),attn_metadata.num_seqs=tensor([1], device='xla:0', dtype=torch.int32),NUM_KV_PAGES_PER_BLOCK=128,NUM_QUERIES_PER_BLOCK=16,attn_metadata.context_lens=tensor([1856,    0,    0,  ...,    0,    0,    0], device='xla:0',
  # dtype=torch.int32),attn_metadata.query_start_loc=tensor([0, 1, 0,  ..., 0, 0, 0], device='xla:0', dtype=torch.int32),query.dtype=torch.bfloat16,key_cache.dtype=torch.bfloat16,value_cache.dtype=torch.bfloat16,attn_metadata.context_lens.dtype=torch.int32,attn_metadata.block_tables.dtype=torch.int32,attn_metadata.query_start_loc.dtype=torch.int32, self.scale=0.08838834764831845
  kv_lens = [1856]
  cu_q_lens = [0, 1]
  num_heads = (32, 8)  # (num_q_heads, num_kv_heads)
  head_dim = 128
  dtype = jnp.bfloat16
  torch_dtype = torch.bfloat16
  page_size = 16
  total_num_pages = 6112
  num_queries_per_block = args.num_queries_per_block
  num_kv_pages_per_block = args.num_kv_pages_per_block
  pages_per_seq=128
  num_seqs = 1
  max_num_batched_tokens = 16
  max_num_reqs = 1024
  vmem_limit_bytes=128000000  # 128 MB on Trillium

  for i in range(num_seqs):
    cur_q_len = cu_q_lens[i+1]-cu_q_lens[i]
    cur_kv_len = kv_lens[i]
    assert cur_q_len <= cur_kv_len, f"cur_q_len must be less than or equal to cur_kv_len. Got {cur_q_len} and {cur_kv_len}"

  # query_lens = [seq_len[0] for seq_len in seq_lens]
  sm_scale = head_dim**-0.5
  # kv_lens = jnp.array([seq_len[1] for seq_len in seq_lens])
  num_q_heads = num_heads[0]
  num_kv_heads = num_heads[1]
  assert num_q_heads % num_kv_heads == 0, "num_q_heads % num_kv_heads !=0."

  prng_key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  queries = jax.random.normal(
      k1, (max_num_batched_tokens, num_q_heads, head_dim), dtype=dtype)
  k_pages = jax.random.normal(
      k2, (total_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k3, (total_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype)

  # Create a kv_lens: i32[max_num_seqs]
  kv_lens_np = jnp.array(kv_lens)
  kv_lens_np = jnp.pad(kv_lens_np, (0, max_num_reqs-len(kv_lens)))

  # Create a page_indices: jax.Array,	# i32[max_num_seqs, pages_per_sequence]
  page_indices = jax.random.randint(
      k4, (max_num_reqs, pages_per_seq), 0, total_num_pages, dtype=jnp.int32)

  # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
  cu_q_lens_np = jnp.array(cu_q_lens)
  cu_q_lens_np = jnp.pad(cu_q_lens_np, (0, max_num_reqs-len(kv_lens)))

  num_seqs_np = jnp.array([num_seqs])

  if _run_with_torch_xla(args.kernel):
    queries_xla = torch.from_numpy(np.array(queries).astype(jnp.float32)).to(
        torch_dtype).to("xla")
    k_pages_xla = torch.from_numpy(np.array(k_pages).astype(jnp.float32)).to(
        torch_dtype).to("xla")
    v_pages_xla = torch.from_numpy(np.array(v_pages).astype(jnp.float32)).to(
        torch_dtype).to("xla")
    kv_lens_xla = torch.from_numpy(np.array(kv_lens_np)).to("xla")
    page_indices_xla = torch.from_numpy(np.array(page_indices)).to("xla")
    cu_q_lens_xla = torch.from_numpy(np.array(cu_q_lens_np)).to("xla")
    num_seqs_xla = torch.from_numpy(np.array(num_seqs_np)).to("xla")

    def ragged_paged_attention_wrapper(q, k_pages, v_pages, kv_lens,
                                       page_indices, cu_q_lens, num_seqs,
                                       num_kv_pages_per_block,
                                       num_queries_per_block, use_kernel, sm_scale, vmem_limit_bytes):
      return torch.ops.xla.ragged_paged_attention(
          q,
          k_pages,
          v_pages,
          kv_lens,
          page_indices,
          cu_q_lens,
          num_seqs,
          num_kv_pages_per_block,
          num_queries_per_block,
          use_kernel=use_kernel,
          sm_scale=sm_scale,
          vmem_limit_bytes=vmem_limit_bytes,
      )

    compiled_paged_attention = torch.compile(
        ragged_paged_attention_wrapper, backend="openxla")

  def run_benchmark(num_iters: int) -> float:
    start_time = time.perf_counter()

    for _ in range(num_iters):
      if args.kernel == "ragged-paged-attention-with-torch-xla-dynamo":
        compiled_paged_attention(
            queries_xla,
            k_pages_xla,
            v_pages_xla,
            kv_lens_xla,
            page_indices_xla,
            cu_q_lens_xla,
            num_seqs_xla,
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
            use_kernel=True,
            sm_scale=sm_scale,
            vmem_limit_bytes=vmem_limit_bytes,
        )
      elif args.kernel == "ragged-paged-attention-with-torch-xla-nondynamo":
        torch.ops.xla.ragged_paged_attention(
            queries_xla,
            k_pages_xla,
            v_pages_xla,
            kv_lens_xla,
            page_indices_xla,
            cu_q_lens_xla,
            num_seqs,
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
            use_kernel=True,
        )
      elif args.kernel == "ragged-paged-attention":
        err, actual_output = ragged_paged_attention(
            queries,
            k_pages,
            v_pages,
            kv_lens_np,
            page_indices,
            cu_q_lens,
            num_seqs,
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
        )
        err.throw()
      elif args.kernel == "ragged-paged-attention-ref-impl":
        actual_output = _ref_ragged_paged_attention(
            queries,
            k_pages,
            v_pages,
            kv_lens_np,
            page_indices,
            cu_q_lens,
            num_seqs,
        )
      else:
        assert False, f"Invalid kernel name {args.kernel}"

    if _run_with_torch_xla(args.kernel):
      xm.mark_step()
      xm.wait_device_ops()
    else:
      jax.block_until_ready(actual_output)

    end_time = time.perf_counter()
    return (end_time - start_time) / num_iters

  # Warmup.
  print("Warming up...")
  run_benchmark(num_iters=3)

  print(
      f"Run benchmark with {num_queries_per_block=}, {num_kv_pages_per_block=} ..."
  )
  latency = run_benchmark(num_iters=10)
  print(f"Kernel running time: {latency * 1000000:.3f} us")


def benchmark_prefill(args):
  # xw32 pallas not allclose query.shape=torch.Size([2048, 32, 128]),key_cache.shape=torch.Size([6112, 16, 8, 128]),value_cache.shape=torch.Size([6112, 16, 8, 128]),attn_metadata.context_lens.shape=torch.Size([1024]),attn_metadata.block_tables.shape=torch.Size([1024, 128]),attn_metadata.query_start_loc.shape=torch.Size([1025]),attn_metadata.num_seqs=tensor([1], device='xla:0', dtype=torch.int32),NUM_KV_PAGES_PER_BLOCK=128,NUM_QUERIES_PER_BLOCK=16,attn_metadata.context_lens=tensor([1853,    0,    0,  ...,    0,    0,    0], device='xla:0',
  # dtype=torch.int32),attn_metadata.query_start_loc=tensor([   0, 1853,    0,  ...,    0,    0,    0], device='xla:0', dtype=torch.int32),query.dtype=torch.bfloat16,key_cache.dtype=torch.bfloat16,value_cache.dtype=torch.bfloat16,attn_metadata.context_lens.dtype=torch.int32,attn_metadata.block_tables.dtype=torch.int32,attn_metadata.query_start_loc.dtype=torch.int32, self.scale=0.08838834764831845
  kv_lens = [1853]
  cu_q_lens = [0, 1853]
  num_heads = (32, 8)  # (num_q_heads, num_kv_heads)
  head_dim = 128
  dtype = jnp.bfloat16
  torch_dtype = torch.bfloat16
  page_size = 16
  total_num_pages = 6112
  num_queries_per_block = args.num_queries_per_block
  num_kv_pages_per_block = args.num_kv_pages_per_block
  pages_per_seq=128
  num_seqs = 1
  max_num_batched_tokens = 2048
  max_num_reqs = 1024
  vmem_limit_bytes=128000000  # 128 MB on Trillium

  for i in range(num_seqs):
    cur_q_len = cu_q_lens[i+1]-cu_q_lens[i]
    cur_kv_len = kv_lens[i]
    assert cur_q_len <= cur_kv_len, f"cur_q_len must be less than or equal to cur_kv_len. Got {cur_q_len} and {cur_kv_len}"

  # query_lens = [seq_len[0] for seq_len in seq_lens]
  sm_scale = head_dim**-0.5
  # kv_lens = jnp.array([seq_len[1] for seq_len in seq_lens])
  num_q_heads = num_heads[0]
  num_kv_heads = num_heads[1]
  assert num_q_heads % num_kv_heads == 0, "num_q_heads % num_kv_heads !=0."

  prng_key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  queries = jax.random.normal(
      k1, (max_num_batched_tokens, num_q_heads, head_dim), dtype=dtype)
  k_pages = jax.random.normal(
      k2, (total_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k3, (total_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype)

  # Create a kv_lens: i32[max_num_seqs]
  kv_lens_np = jnp.array(kv_lens)
  kv_lens_np = jnp.pad(kv_lens_np, (0, max_num_reqs-len(kv_lens)))

  # Create a page_indices: jax.Array,	# i32[max_num_seqs, pages_per_sequence]
  page_indices = jax.random.randint(
      k4, (max_num_reqs, pages_per_seq), 0, total_num_pages, dtype=jnp.int32)

  # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
  cu_q_lens_np = jnp.array(cu_q_lens)
  cu_q_lens_np = jnp.pad(cu_q_lens_np, (0, max_num_reqs-len(kv_lens)))

  num_seqs_np = jnp.array([num_seqs])

  if _run_with_torch_xla(args.kernel):
    queries_xla = torch.from_numpy(np.array(queries).astype(jnp.float32)).to(
        torch_dtype).to("xla")
    k_pages_xla = torch.from_numpy(np.array(k_pages).astype(jnp.float32)).to(
        torch_dtype).to("xla")
    v_pages_xla = torch.from_numpy(np.array(v_pages).astype(jnp.float32)).to(
        torch_dtype).to("xla")
    kv_lens_xla = torch.from_numpy(np.array(kv_lens_np)).to("xla")
    page_indices_xla = torch.from_numpy(np.array(page_indices)).to("xla")
    cu_q_lens_xla = torch.from_numpy(np.array(cu_q_lens_np)).to("xla")
    num_seqs_xla = torch.from_numpy(np.array(num_seqs_np)).to("xla")

    def ragged_paged_attention_wrapper(q, k_pages, v_pages, kv_lens,
                                       page_indices, cu_q_lens, num_seqs,
                                       num_kv_pages_per_block,
                                       num_queries_per_block, use_kernel, sm_scale, vmem_limit_bytes):
      return torch.ops.xla.ragged_paged_attention(
          q,
          k_pages,
          v_pages,
          kv_lens,
          page_indices,
          cu_q_lens,
          num_seqs,
          num_kv_pages_per_block,
          num_queries_per_block,
          use_kernel=use_kernel,
          sm_scale=sm_scale,
          vmem_limit_bytes=vmem_limit_bytes,
      )

    compiled_paged_attention = torch.compile(
        ragged_paged_attention_wrapper, backend="openxla")

  def run_benchmark(num_iters: int) -> float:
    start_time = time.perf_counter()

    for _ in range(num_iters):
      if args.kernel == "ragged-paged-attention-with-torch-xla-dynamo":
        compiled_paged_attention(
            queries_xla,
            k_pages_xla,
            v_pages_xla,
            kv_lens_xla,
            page_indices_xla,
            cu_q_lens_xla,
            num_seqs_xla,
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
            use_kernel=True,
            sm_scale=sm_scale,
            vmem_limit_bytes=vmem_limit_bytes,
        )
      elif args.kernel == "ragged-paged-attention-with-torch-xla-nondynamo":
        torch.ops.xla.ragged_paged_attention(
            queries_xla,
            k_pages_xla,
            v_pages_xla,
            kv_lens_xla,
            page_indices_xla,
            cu_q_lens_xla,
            num_seqs,
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
            use_kernel=True,
        )
      elif args.kernel == "ragged-paged-attention":
        err, actual_output = ragged_paged_attention(
            queries,
            k_pages,
            v_pages,
            kv_lens_np,
            page_indices,
            cu_q_lens,
            num_seqs,
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
        )
        err.throw()
      elif args.kernel == "ragged-paged-attention-ref-impl":
        actual_output = _ref_ragged_paged_attention(
            queries,
            k_pages,
            v_pages,
            kv_lens_np,
            page_indices,
            cu_q_lens,
            num_seqs,
        )
      else:
        assert False, f"Invalid kernel name {args.kernel}"

    if _run_with_torch_xla(args.kernel):
      xm.mark_step()
      xm.wait_device_ops()
    else:
      jax.block_until_ready(actual_output)

    end_time = time.perf_counter()
    return (end_time - start_time) / num_iters

  # Warmup.
  print("Warming up...")
  run_benchmark(num_iters=3)

  print(
      f"Run benchmark with {num_queries_per_block=}, {num_kv_pages_per_block=} ..."
  )
  latency = run_benchmark(num_iters=10)
  print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--kernel",
      type=str,
      choices=[
          "ragged-paged-attention",
          "ragged-paged-attention-with-torch-xla-dynamo",
          "ragged-paged-attention-with-torch-xla-nondynamo",
          "ragged-paged-attention-ref-impl",
      ],
      default="multi-queries-paged-attn")
  parser.add_argument("--num-queries-per-block", type=int, default=128)
  parser.add_argument("--num-kv-pages-per-block", type=int, default=128)
  args = parser.parse_args()

  # benchmark_prefill(args)
  benchmark_decode(args)
