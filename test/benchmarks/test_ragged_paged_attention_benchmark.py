# Usage: python pytorch/xla/test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged-paged-attention

import argparse
import time
from typing import List, Optional, Tuple
import functools

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention as jax_single_query_paged_attention
import jax.numpy as jnp
import numpy as np

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


def _get_closest_power_of_two(x):
  if x <= 0:
    raise ValueError(f"x must be positive. Got {x}")
  return 2**int(np.ceil(np.log2(x)))


def benchmark(args):
  seq_lens = [
      (1, 1328),
      (5, 18),
      (1, 129),
      (120, 229),
      (1, 122),  # end of the first physical q block
      (1, 64),
      (32, 100),
      (250, 463),
      (1, 18),
      (1, 17),
      (99, 123),  # last 3 physical q blocks [(q_len, kv_len),...]
  ]
  num_heads = (4, 4)
  head_dim = 128
  dtype = jnp.float32
  page_size = 16
  num_pages = 32768
  num_queries_per_block = 128

  num_seqs = len(seq_lens)
  # Make sure the q_len is no longer than the kv_len. For example,
  # seq_lens = [(1, 1328), (5, 18), (506, 463)] is not a valid test case because
  # the 3rd sequence has q_len(506) > kv_len(463).
  for i in range(num_seqs):
    cur_q_len = seq_lens[i][0]
    cur_kv_len = seq_lens[i][1]
    assert cur_q_len <= cur_kv_len, f"cur_q_len must be less than or equal to cur_kv_len. Got {cur_q_len} and {cur_kv_len}"

  query_lens = [seq_len[0] for seq_len in seq_lens]
  num_q_tokens = sum(query_lens)
  kv_lens = jnp.array([seq_len[1] for seq_len in seq_lens])
  num_q_heads = num_heads[0]
  num_kv_heads = num_heads[1]
  assert num_q_heads % num_kv_heads == 0, "num_q_heads % num_kv_heads !=0."

  prng_key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  queries = jax.random.normal(
      k1, (num_q_tokens, num_q_heads, head_dim), dtype=dtype)
  k_pages = jax.random.normal(
      k2, (num_kv_heads, num_pages, page_size, head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k3, (num_kv_heads, num_pages, page_size, head_dim), dtype=dtype)

  # Create a kv_lens: i32[num_tokens]
  kv_lens_with_paddings = [0] * num_q_tokens
  for i in range(num_seqs):
    kv_lens_with_paddings[i] = kv_lens[i]
  kv_lens_np = jnp.array(kv_lens_with_paddings)

  # Create a page_indices: jax.Array,	# i32[num_tokens, pages_per_sequence]
  max_kv_len = max([seq_len[1] for seq_len in seq_lens])
  max_num_pages_per_seq = (max_kv_len + page_size - 1) // page_size
  # The reason why we need to pad max_num_pages_per_seq is that
  # page_indices[1]=max_num_pages_per_seq and max_num_pages_per_seq%num_kv_pages_per_compute_block==0
  max_num_pages_per_seq = _get_closest_power_of_two(max_num_pages_per_seq)
  # The assert below mimics the reality that each page get a unique index.
  # But for testing, the assert could be omitted.
  # assert max_num_pages_per_seq*num_q_tokens <= num_pages, f"assert failed: max_num_pages_per_seq*num_q_tokens < num_pages. Got {max_num_pages_per_seq*num_q_tokens} and {num_pages}"
  page_indices = jax.random.randint(
      k4, (num_q_tokens, max_num_pages_per_seq), 0, num_pages, dtype=jnp.int32)

  # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
  q_lens_with_paddings = [0] * num_q_tokens
  for i in range(num_seqs):
    q_lens_with_paddings[i] = query_lens[i]
  cu_q_lens = jnp.cumsum(jnp.array([0] + q_lens_with_paddings))

  err, actual_output = ragged_paged_attention(
      queries,
      k_pages,
      v_pages,
      kv_lens_np,
      page_indices,
      cu_q_lens,
      num_seqs,
      num_queries_per_block=num_queries_per_block,
  )
  err.throw()  # noop if there is no error.
  actual_output = jax.block_until_ready(actual_output)
  profile_path = "/workspaces/persist/myprofiles/plugins/profile"

  def run_benchmark(num_iters: int, profile: bool = False) -> float:
    start_time = time.perf_counter()
    if profile:
      jax.profiler.start_trace(profile_path)

    actual_output = None
    for _ in range(num_iters):
      if args.kernel == "ragged-paged-attention":
        err, actual_output = ragged_paged_attention(
            queries,
            k_pages,
            v_pages,
            kv_lens_np,
            page_indices,
            cu_q_lens,
            num_seqs,
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
    latency = run_benchmark(num_iters=10, profile=False)
  print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--kernel",
      type=str,
      choices=[
          "ragged-paged-attention",
          "ragged-paged-attention-ref-impl",
      ],
      default="multi-queries-paged-attn")
  parser.add_argument("--profile", action="store_true")
  args = parser.parse_args()
  benchmark(args)
