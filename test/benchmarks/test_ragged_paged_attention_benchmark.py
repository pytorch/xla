# For benchmarking prefill without dynamo: python test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged_paged_attention-v2 --case decode
# For benchmarking decode without dynamo: python test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged_paged_attention-v2 --case decode

# Usage: python test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged-paged-attention-with-torch-xla-dynamo
# Usage: python test/benchmarks/test_ragged_paged_attention_benchmark.py --kernel ragged_paged_attention-v2
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
  from torch_xla.experimental.pallas_kernels.ragged_paged_attention_v2 import ragged_paged_attention as ragged_paged_attention_v2
  from torch_xla.experimental.pallas_kernels.ragged_paged_attention_v2 import validate_inputs_on_runtime


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

def _get_common_kernel_input(num_kv_pages_per_block):
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

  num_seqs = len(seq_lens)
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
      k2, (num_pages, page_size, num_kv_heads * head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k3, (num_pages, page_size, num_kv_heads *head_dim), dtype=dtype)

  # Create a kv_lens: i32[num_tokens]
  kv_lens_with_paddings = [0] * num_q_tokens
  kv_lens_with_paddings[:num_seqs] = kv_lens[:num_seqs]
  kv_lens_np = jnp.array(kv_lens_with_paddings)

  # Create a page_indices: jax.Array,	# i32[num_tokens, pages_per_sequence]
  max_kv_len = max([seq_len[1] for seq_len in seq_lens])
  max_num_pages_per_seq = (max_kv_len + page_size - 1) // page_size
  # The reason why we need to pad max_num_pages_per_seq is that
  # page_indices[1]=max_num_pages_per_seq and max_num_pages_per_seq%num_kv_pages_per_compute_block==0
  max_num_pages_per_seq = _get_closest_of_multiple(max_num_pages_per_seq,
                                                   num_kv_pages_per_block)
  page_indices = jax.random.randint(
      k4, (num_q_tokens, max_num_pages_per_seq), 0, num_pages, dtype=jnp.int32)

  # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
  q_lens_with_paddings = [0] * num_q_tokens
  for i in range(num_seqs):
    q_lens_with_paddings[i] = query_lens[i]
  cu_q_lens = jnp.cumsum(jnp.array([0] + q_lens_with_paddings))
  return queries, k_pages, v_pages, kv_lens_np, page_indices, cu_q_lens, num_seqs

def _get_prefill_kernel_input():
  prng_key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  num_q_tokens = 512
  max_num_seqs = 512
  num_q_heads = 32
  head_dim = 128
  # torch.from_numpy cannot convert bf16
  dtype = jnp.float32
  queries = jax.random.normal(
      k1, (num_q_tokens, num_q_heads, head_dim), dtype=dtype)
  
  num_kv_heads = 8
  num_pages = 6140
  page_size = 16
  k_pages = jax.random.normal(
      k2, (num_pages, page_size, num_kv_heads * head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k3, (num_pages, page_size, num_kv_heads *head_dim), dtype=dtype)

  kv_lens = jnp.array([1845, 1831, 1826, 1833, 1832, 1794, 1808, 1780, 1791, 1781, 1820, 1750, 1799, 1815, 1319,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0], dtype=jnp.int32)
  pages_per_seq = 128
  page_indices = jax.random.randint(
      k4, (max_num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
  cu_q_lens=jnp.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 14, 512,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 1,   1,   1,   1,   1,   1,   1,   1,   1], dtype=jnp.int32)
  num_seqs = 15
  return queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens, num_seqs

def _get_decode_kernel_input():
  prng_key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  num_q_tokens = 16
  max_num_seqs = 512
  num_q_heads = 32
  head_dim = 128
  # torch.from_numpy cannot convert bf16
  dtype = jnp.float32
  queries = jax.random.normal(
      k1, (num_q_tokens, num_q_heads, head_dim), dtype=dtype)
  
  num_kv_heads = 8
  num_pages = 6088
  page_size = 16
  k_pages = jax.random.normal(
      k2, (num_pages, page_size, num_kv_heads * head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k3, (num_pages, page_size, num_kv_heads *head_dim), dtype=dtype)

  kv_lens = jnp.array([1877, 1894, 1924, 1883, 1909, 1914, 1895, 1887, 1913, 1894, 1884, 1876, 1879, 1942, 1908, 1920, 1911, 1915, 1876, 1912, 1910, 1901, 1910, 1895, 1881, 1888, 1895, 1880, 1881, 1875, 1833, 1864, 1853, 1854, 1871, 1835, 1042,   22,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0], dtype=jnp.int32)
  pages_per_seq = 128
  # NB, the randint upper bound `num_pages` need to align with kv_cache.shape[0] (aka num_pages). Otherwise, kernel will fail with `jaxlib.xla_extension.XlaRuntimeError: FAILED_PRECONDITION: The program continuator has halted unexpectedly` meaning a out-of-bound indexing error.
  page_indices = jax.random.randint(
      k4, (max_num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
  cu_q_lens=jnp.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=jnp.int32)
  num_seqs = 11
  return queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens, num_seqs

def benchmark(args):
  print(f'Running benchmark on {args.kernel=}, {args.case=}')
  num_queries_per_block = args.num_queries_per_block
  num_kv_pages_per_block = args.num_kv_pages_per_block
  if args.case == "prefill":
    queries, k_pages, v_pages, kv_lens_np, page_indices, cu_q_lens, num_seqs = _get_prefill_kernel_input()
  elif args.case == "decode":
    queries, k_pages, v_pages, kv_lens_np, page_indices, cu_q_lens, num_seqs = _get_decode_kernel_input()
  else:
    queries, k_pages, v_pages, kv_lens_np, page_indices, cu_q_lens, num_seqs = _get_common_kernel_input(num_kv_pages_per_block)

  validate_inputs_on_runtime(queries, k_pages, v_pages, kv_lens_np, page_indices, cu_q_lens, jnp.array([  num_seqs]))
  
  target_jax_dtype = jnp.bfloat16
  target_torch_dtype = torch.bfloat16

  if _run_with_torch_xla(args.kernel):
    queries_xla = torch.from_numpy(np.array(queries)).to(
        target_torch_dtype).to("xla")
    k_pages_xla = torch.from_numpy(np.array(k_pages)).to(
        target_torch_dtype).to("xla")
    v_pages_xla = torch.from_numpy(np.array(v_pages)).to(
        target_torch_dtype).to("xla")
    kv_lens_xla = torch.from_numpy(np.array(kv_lens_np)).to("xla")
    page_indices_xla = torch.from_numpy(np.array(page_indices)).to("xla")
    cu_q_lens_xla = torch.from_numpy(np.array(cu_q_lens)).to("xla")
    num_seqs_xla = torch.tensor([num_seqs], dtype=torch.int32).to("xla")

    def ragged_paged_attention_wrapper(q, k_pages, v_pages, kv_lens,
                                       page_indices, cu_q_lens, num_seqs,
                                       num_kv_pages_per_block,
                                       num_queries_per_block, use_kernel):
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
      )

    compiled_paged_attention = torch.compile(
        ragged_paged_attention_wrapper, backend="openxla")

  queries.astype(target_jax_dtype)
  k_pages.astype(target_jax_dtype)
  v_pages.astype(target_jax_dtype)
  # print(f'{queries.shape=}, {k_pages.shape=}, {v_pages.shape=}, {kv_lens_np.shape=}, {page_indices.shape=}, {cu_q_lens.shape=}, {num_seqs=}, {queries.dtype=}, {k_pages.dtype=}, {v_pages.dtype=}, {page_indices=}')

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
        )
      elif args.kernel == "ragged-paged-attention-with-torch-xla-nondynamo":
        torch.ops.xla.ragged_paged_attention(
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
      elif args.kernel == "ragged_paged_attention-v2":
        actual_output = ragged_paged_attention_v2(
            queries,
            k_pages,
            v_pages,
            kv_lens_np,
            page_indices,
            cu_q_lens,
            jnp.array([num_seqs]),
            num_queries_per_block=num_queries_per_block,
            num_kv_pages_per_block=num_kv_pages_per_block,
        )
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
          "ragged_paged_attention-v2",
          "ragged-paged-attention-with-torch-xla-dynamo",
          "ragged-paged-attention-with-torch-xla-nondynamo",
          "ragged-paged-attention-ref-impl",
      ],
      default="multi-queries-paged-attn")
  parser.add_argument("--num-queries-per-block", type=int, default=128)
  parser.add_argument("--num-kv-pages-per-block", type=int, default=128)
  parser.add_argument("--case",
                      type=str,
                      choices=[
                        "prefill",
                        "decode",
                      ],
                      default="")
  args = parser.parse_args()

  benchmark(args)
