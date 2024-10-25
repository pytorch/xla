# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention.extended_paged_attention_kernel1 import paged_attention as jax_extended_paged_attention1
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


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
):
  assert max_kv_len % page_size == 0
  pages_per_sequence = max_kv_len // page_size
  batch_size = len(kv_seq_lens)
  total_pages = batch_size * pages_per_sequence
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(k4, (batch_size, query_len, num_q_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices


# Prepare ref impl.
def _reconstruct_kv(page_indices, kv_pages):
  batch_size = page_indices.shape[0]
  num_kv_heads, _, _, head_dim = kv_pages.shape

  def per_sequence_page_gather(kv_pages, page_indices):
    return jnp.take(kv_pages, page_indices, 1)

  gathered = jax.vmap(per_sequence_page_gather, in_axes=(None, 0))(
      kv_pages, page_indices
  )
  return gathered.reshape(batch_size, num_kv_heads, -1, head_dim)


def _grouped_query_attention_reference(q, k, v, lengths, attn_logits_soft_cap):
  batch_size, num_heads, head_dim = q.shape
  _, num_kv_heads, max_seq_len, _ = k.shape
  assert k.shape == v.shape
  assert num_heads % num_kv_heads == 0
  q = q.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)

  if isinstance(k, quantization_utils.QuantizedTensor):
    k = quantization_utils.unquantize_from_int8(k, dtype=jnp.float32)
  if isinstance(v, quantization_utils.QuantizedTensor):
    v = quantization_utils.unquantize_from_int8(v, dtype=jnp.float32)

  logits = jnp.einsum(
      "bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap
  mask = jnp.arange(max_seq_len)[None] < lengths[:, None]
  mask_value = -0.7 * float(np.finfo(np.dtype("float32")).max)
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, :]
  weights = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("bhgt,bhtd->bhgd", weights.astype(v.dtype), v)
  return o.reshape(batch_size, num_heads, head_dim)


def _megacore_enabled():
  return jax.devices()[0].device_kind == "TPU v4" or jtu.is_device_tpu(
      version=5, variant="p"
  )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    print(f'{jax.devices()[0].device_kind=}')
    if not jtu.is_device_tpu(
      version=5, variant="e"):
      self.skipTest("Only run on v5e")

  # @parameterized.product(
  #     dtype=(jnp.float32, jnp.bfloat16),
  #     page_size=(16, 32, 64),
  #     num_kv_heads=(1, 8),
  #     q_kv_head_ratio=(1, 4, 8),
  #     head_dim=(128, 256),
  #     megacore_mode=("batch", "kv_head", None),
  #     attn_logits_soft_cap=(1.0, None),
  #     are_kv_quantized=(
  #         False,
  #         True,
  #     ),
  # )
  # @parameterized.product(
  #     dtype=(jnp.float32,),
  #     page_size=(16,),
  #     num_kv_heads=(1,),
  #     q_kv_head_ratio=(1,),
  #     head_dim=(128,),
  #     megacore_mode=(None,),
  #     attn_logits_soft_cap=(None,),
  #     are_kv_quantized=(
  #         False,
  #     ),
  # )
  # def test_paged_attention(
  #     self,
  #     dtype,
  #     page_size,
  #     num_kv_heads,
  #     q_kv_head_ratio,
  #     head_dim,
  #     megacore_mode,
  #     attn_logits_soft_cap,
  #     are_kv_quantized,
  # ):
  #   if not jtu.is_device_tpu_at_least(4):
  #     self.skipTest("Only supports TPU generation 4 or above")
  #   if jtu.is_device_tpu(version=4) and are_kv_quantized:
  #     # TPU v4 has only 16MiB of VMEM which is not sufficient to store both the
  #     # weight and scale tensors for quantized tensors. When enabled on TPUv4,
  #     # the tests sometimes failed with resource exhausted error.
  #     self.skipTest("Quantization is not supported on TPU v4")
  #   if megacore_mode and not _megacore_enabled():
  #     self.skipTest("Megacore is only available on TPU v4 or TPU v5p")
  #   if num_kv_heads % 2 != 0 and megacore_mode == "kv_head":
  #     self.skipTest("Skip kv_head megacore mode when num_kv_heads is odd")
  #   max_kv_len = 2048
  #   block_size = 512
  #   seq_lens = np.asarray([0, 3, 256, 513, 1023, 2048])
  #   q, k_pages, v_pages, page_indices = _generate_qkv(
  #       seq_lens,
  #       page_size,
  #       max_kv_len,
  #       num_kv_heads,
  #       num_kv_heads * q_kv_head_ratio,
  #       head_dim,
  #       jax.random.key(0),
  #       dtype,
  #       are_kv_quantized=are_kv_quantized,
  #   )
  #   o = paged_attention.paged_attention(
  #       q,
  #       k_pages,
  #       v_pages,
  #       seq_lens,
  #       page_indices,
  #       pages_per_compute_block=block_size // page_size,
  #       megacore_mode=megacore_mode,
  #       attn_logits_soft_cap=attn_logits_soft_cap,
  #   )
  #   k = _reconstruct_kv(page_indices, k_pages)
  #   v = _reconstruct_kv(page_indices, v_pages)
  #   o_ref = _grouped_query_attention_reference(
  #       q, k, v, seq_lens, attn_logits_soft_cap)

  #   if q_kv_head_ratio > 1:
  #     atol, rtol = 1e-2, 2e-2
  #   else:
  #     atol, rtol = 2e-1, 1e-1
  #   np.testing.assert_allclose(
  #       o[np.where(seq_lens > 0)].astype(jnp.float32),
  #       o_ref[np.where(seq_lens > 0)].astype(jnp.float32),
  #       atol=atol,
  #       rtol=rtol,
  #   )

  def _ref_jax_extended_paged_attention(
      self,
      q,      # [batch_size, query_len, num_query_heads, head_size]
      k_pages,# [num_kv_heads, total_num_pages, page_size, head_size]
      v_pages,# [num_kv_heads, total_num_pages, page_size, head_size]
      lengths,# [batch_size]
      page_indices,# [batch_size, pages_per_sequence]
  ):
    batch_size, query_len, num_query_heads, head_size = q.shape
    num_kv_heads, total_num_pages, page_size, _ = k_pages.shape
    num_query_per_kv = num_query_heads // num_kv_heads

    lengths = lengths
    page_indices = page_indices
    outputs = []
    for i in range(batch_size):
      kv_len = lengths[i]
      num_pages = (kv_len+page_size-1) // page_size
      indices = page_indices[i, :num_pages]
      print(f'xw32 indices={indices}')

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
      q_span = (kv_len-query_len) + jax.lax.broadcasted_iota(
          jnp.int32, (query_len, kv_len), 0
      )
      kv_span = jax.lax.broadcasted_iota(
          jnp.int32, (query_len, kv_len), 1
      )
      mask=jnp.where(q_span < kv_span, float("-inf"), 0.)
      with jax.numpy_rank_promotion("allow"):
        attn = attn + mask
      attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
      out = jnp.einsum("hqk,khd->qhd", attn, v)
      outputs.append(out)
    output = jnp.stack(outputs, axis=0)
    return output

  def test_extended_paged_attention_v1_multiple_queries(self):
    # num_compute_blks_q=1, num_compute_blks_kv=1,num_q_heads_per_kv_head=1,
    # num_compute_blks_q: =(query_len//num_queries_per_compute_block)
    # Change num_queries_per_compute_block to adjust num_compute_blks_q
    # num_compute_blks_kv: =(pages_per_sequence//num_kv_pages_per_compute_block) where
    # pages_per_sequence=max_kv_seq_len // page_size=32, num_kv_pages_per_compute_block=block_kv_size//page_size.
    # So num_compute_blks_kv=max_kv_len//block_kv_size
    # Change pallas_compute_block_size to adjust num_compute_blks_kv
    # Change num_query_heads to adjust num_q_heads_per_kv_head.
    dtype = jnp.float32  # (jnp.float32, jnp.bfloat16)
    page_size = 64  # (16, 32, 64)
    num_kv_heads = 1  # (1, 8)
    q_kv_head_ratio=1  # (1, 4, 8)
    head_dim=128  # (128, 256)
    max_kv_len = 512  # 2048
    block_kv_size = 512
    num_queries_per_compute_block = 16
    kv_seq_lens = jnp.asarray([64])  # np.asarray([0, 3, 256, 513, 1023, 2048])
    query_len = 16

    assert query_len <= max_kv_len
    batch_size = len(kv_seq_lens)
    pages_per_sequence = max_kv_len // page_size
    total_num_pages = batch_size * pages_per_sequence
    assert max_kv_len <= total_num_pages * page_size

    q, k_pages, v_pages, page_indices = _generate_qkv(
        kv_seq_lens,
        page_size,
        max_kv_len,
        query_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
    )
    # Run the extended_paged_attention with query_len>1
    print(f'xw32 calling jax_extended_paged_attention1, {query_len=}')
    num_kv_pages_per_compute_block=block_kv_size // page_size
    actual_output = jax_extended_paged_attention1(
                 q,
                 k_pages,
                 v_pages,
                 kv_seq_lens,
                 page_indices,
                 num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
                 num_queries_per_compute_block=num_queries_per_compute_block,
             )
    actual_output = jax.block_until_ready(actual_output)
    print('my new extended paged attention finished yay')

    # Run Woosuk's non-kernel impl but in the JAX version that I wrote.
    expected_output = self._ref_jax_extended_paged_attention(
      q,
      k_pages,
      v_pages,
      kv_seq_lens,
      page_indices,
    )

    expected_output_cpu=expected_output
    actual_output_cpu=actual_output
    # print(f'{expected_output_cpu=}')
    # print(f'{actual_output_cpu=}')
    # print(f'actual_output_cpu.shape={actual_output_cpu.shape}')
    # print(f'expected_output_cpu.shape={expected_output_cpu.shape}')
    self.assertEqual(actual_output_cpu.shape, expected_output_cpu.shape)
    # torch.set_printoptions(profile="full")
    # print(f'{jnp.abs(actual_output_cpu-expected_output_cpu)}')
    print(f'Output max diff: {jnp.max(jnp.abs(expected_output_cpu - actual_output_cpu))}')
    print(f'Output mean diff: {jnp.mean(jnp.abs(expected_output_cpu - actual_output_cpu))}')

    self.assertTrue(
        jnp.allclose(
            expected_output_cpu[:,:,0],
            actual_output_cpu[:,:,0],
            atol=1e-2,
            rtol=1e-2))
    self.assertTrue(
        jnp.allclose(
            expected_output_cpu,
            actual_output_cpu,
            atol=1e-2,
            rtol=1e-2))

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
