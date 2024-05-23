import time

from absl.testing import absltest
from absl.testing import parameterized

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

from torch_xla.experimental.custom_kernel import paged_attention as torch_xla_paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention as jax_paged_attention

import jax
import jax.numpy as jnp
import numpy as np


def _paged_attention_xla(q, k, v, seq_lens, page_indices, block_size, page_size, megacore_mode):
  start_time = time.time()
  output = torch_xla_paged_attention(
          q,
          k,
          v,
          seq_lens,
          page_indices,
          pages_per_compute_block=block_size // page_size,
          megacore_mode=megacore_mode,
  )
  end_time = time.time()
  # print(f'PyTorch/XLA Time: {end_time - start_time}')
  return end_time - start_time, output

def _paged_attention_jax(q, k, v, seq_lens, page_indices, block_size, page_size, megacore_mode):
  start_time = time.time()
  output = jax_paged_attention(
          q,
          k,
          v,
          seq_lens,
          page_indices,
          pages_per_compute_block=block_size // page_size,
          megacore_mode=megacore_mode,
  )
  end_time = time.time()
  # print(f'JAX Time: {end_time - start_time}')
  return end_time - start_time, output

def _pagedattention_generate_qkv(
    seq_lens,
    page_size,
    max_seq_len,
    num_kv_heads,
    num_heads,
    head_dim,
    dtype=torch.float32,
):
  assert max_seq_len % page_size == 0
  pages_per_sequence = max_seq_len // page_size
  batch_size = len(seq_lens)
  total_pages = batch_size * pages_per_sequence
  k_pages = torch.randn(
      num_kv_heads, total_pages, page_size, head_dim, dtype=dtype)
  v_pages = torch.randn(
      num_kv_heads, total_pages, page_size, head_dim, dtype=dtype)
  page_indices = torch.randperm(
      batch_size * pages_per_sequence, dtype=torch.int32)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
  return q, k_pages, v_pages, page_indices

class BenchmarkAttention(parameterized.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @parameterized.product(
      page_size=(32, 64),
      num_kv_heads=(1, 8),
      q_kv_head_ratio=(4, 8),
      head_dim=(128, 256),
      megacore_mode=("batch", "kv_head", None),
  )
  def test_paged_attention(
      self,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      megacore_mode,
  ):
    if num_kv_heads % 2 != 0 and megacore_mode == "kv_head":
      self.skipTest("Skip kv_head megacore mode when num_kv_heads is odd")

    dtype = torch.float32
    max_kv_len = 2048
    block_size = 512
    seq_lens = torch.tensor([0, 3, 256, 513, 1023, 2048], dtype=torch.int32)
    q, k_pages, v_pages, page_indices = _pagedattention_generate_qkv(
        seq_lens,
        page_size,
        max_kv_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        dtype=dtype,
    )

    q_xla = q.to("xla")
    k_pages_xla = k_pages.to("xla")
    v_pages_xla = v_pages.to("xla")
    seq_lens_xla = seq_lens.to("xla")
    page_indices_xla = page_indices.to("xla")

    q_jax = jnp.array(q.numpy(), dtype=jnp.float32)
    k_pages_jax = jnp.array(k_pages.numpy(), dtype=jnp.float32)
    v_pages_jax = jnp.array(v_pages.numpy(), dtype=jnp.float32)
    seq_lens_jax = jnp.array(seq_lens.numpy(), dtype=jnp.int32)
    page_indices_jax = jnp.array(page_indices.numpy(), dtype=jnp.int32)

    torch_xla_time, _ = _paged_attention_xla(
      q_xla,
      k_pages_xla,
      v_pages_xla,
      seq_lens_xla,
      page_indices_xla,
      block_size,
      page_size,
      megacore_mode=megacore_mode
    )

    jax_time, _ = _paged_attention_jax(
      q_jax,
      k_pages_jax,
      v_pages_jax,
      seq_lens_jax,
      page_indices_jax,
      block_size,
      page_size,
      megacore_mode=megacore_mode
    )

    text = ''
    diff = (jax_time - torch_xla_time) / torch_xla_time
    if torch_xla_time > jax_time:
      text = f'XLA is {abs(diff):.2f}% slower'
    else:
      text = f'XLA is {abs(diff):.2f}% faster'

    print(f'''\
----------
- XLA: {torch_xla_time:.2f}
- JAX: {jax_time:.2f}
{text}
----------
'''
    )


if __name__ == '__main__':
  absltest.main()

