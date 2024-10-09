# DO NOT REVIEW.
# This is the test for the original paged attention in JAX.
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


def _generate_qkv(
    seq_lens, # [0, 3, 256, 513, 1023, 2048]
    page_size, # 16
    max_seq_len, # 2048
    num_kv_heads, # 8
    num_heads, # 64
    head_dim, # 128
    prng_key,
    dtype=jnp.float32,
    are_kv_quantized=False,
):
  assert max_seq_len % page_size == 0
  pages_per_sequence = max_seq_len // page_size # 128
  batch_size = len(seq_lens) # 6
  total_pages = batch_size * pages_per_sequence # 768
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )

  if are_kv_quantized:
    k_pages = quantization_utils.quantize_to_int8(k_pages)
    v_pages = quantization_utils.quantize_to_int8(v_pages)

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(k4, (batch_size, num_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices


def _reconstruct_kv(page_indices, pages):
  # page_indices.shape=i32[batch_size, pages_per_sequence]=(6, 128)
  # pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]=(8, 768, 16, 128)
  if isinstance(pages, quantization_utils.QuantizedTensor):
    pages = quantization_utils.unquantize_from_int8(pages, dtype=jnp.float32)

  batch_size = page_indices.shape[0]
  num_heads, _, _, head_dim = pages.shape # this is really num_kv_heads

  def per_sequence_page_gather(pages, page_indices):
    return jnp.take(pages, page_indices, 1)

  gathered = jax.vmap(per_sequence_page_gather, in_axes=(None, 0))(
      pages, page_indices
  )
  # gathered.reshape(batch_size, num_kv_heads, -1, head_dim).shape=(6, 8, 2048, 128)=[batch_size,num_kv_heads,-1,head_dim]
  # 2048 might be the max_kv_len
  return gathered.reshape(batch_size, num_heads, -1, head_dim)


def _grouped_query_attention_reference(q, k, v, lengths, attn_logits_soft_cap):
  import pdb; pdb.set_trace()
  batch_size, num_heads, head_dim = q.shape
  # k,v: =[batch_size,num_kv_heads,max_kv_len,head_dim]=(6, 8, 2048, 128)
  _, num_kv_heads, max_seq_len, _ = k.shape
  assert k.shape == v.shape
  assert num_heads % num_kv_heads == 0
  q = q.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)

  if isinstance(k, quantization_utils.QuantizedTensor):
    k = quantization_utils.unquantize_from_int8(k, dtype=jnp.float32)
  if isinstance(v, quantization_utils.QuantizedTensor):
    v = quantization_utils.unquantize_from_int8(v, dtype=jnp.float32)

  # q.shape=(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)
  # k.shape=[batch_size,num_kv_heads,max_kv_len,head_dim]=(6, 8, 2048, 128)
  logits = jnp.einsum(
      "bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap
  # if max_seq_len=3, jnp.arange(3)[None]=[[0, 1, 2]]
  # Here max_seq_len=2048, lengths.shape=(6,)
  # lengths=array([   0,    3,  256,  513, 1023, 2048])
  # lengths[:, None]=
  #   array([[   0],
  #        [   3],
  #        [ 256],
  #        [ 513],
  #        [1023],
  #        [2048]])
  mask = jnp.arange(max_seq_len)[None] < lengths[:, None]
  # mask.shape=(6, 2048)
  # (Pdb) p mask
  # Array([[False, False, False, ..., False, False, False],
  #        [ True,  True,  True, ..., False, False, False],
  #        [ True,  True,  True, ..., False, False, False],
  #        [ True,  True,  True, ..., False, False, False],
  #        [ True,  True,  True, ..., False, False, False],
  #        [ True,  True,  True, ...,  True,  True,  True]], dtype=bool)
  mask_value = -0.7 * float(np.finfo(np.dtype("float32")).max)
  # logits.shape=(6, 8, 8, 2048)=[batch_size,num_kv_heads,num_heads // num_kv_heads,max_kv_len]
  # jnp.where(mask, 0.0, mask_value).shape=(6,2048)
  # jnp.where(mask, 0.0, mask_value)[:, None, None, :].shape=(6,1,1,2048)
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
    if jtu.is_device_tpu_at_least(6):
      self.skipTest('Not implemented for TPU v6')

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
  def test_paged_attention(
      self,
      # dtype,
      # page_size,
      # num_kv_heads,
      # q_kv_head_ratio,
      # head_dim,
      # megacore_mode,
      # attn_logits_soft_cap,
      # are_kv_quantized,
  ):
    dtype = jnp.float32
    page_size=16
    num_kv_heads=8
    q_kv_head_ratio=8
    head_dim=128
    megacore_mode=None
    attn_logits_soft_cap, are_kv_quantized = None, False
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supports TPU generation 4 or above")
    if jtu.is_device_tpu(version=4) and are_kv_quantized:
      # TPU v4 has only 16MiB of VMEM which is not sufficient to store both the
      # weight and scale tensors for quantized tensors. When enabled on TPUv4,
      # the tests sometimes failed with resource exhausted error.
      self.skipTest("Quantization is not supported on TPU v4")
    if megacore_mode and not _megacore_enabled():
      self.skipTest("Megacore is only available on TPU v4 or TPU v5p")
    if num_kv_heads % 2 != 0 and megacore_mode == "kv_head":
      self.skipTest("Skip kv_head megacore mode when num_kv_heads is odd")
    max_kv_len = 2048
    block_size = 512
    seq_lens = np.asarray([0, 3, 256, 513, 1023, 2048])
    q, k_pages, v_pages, page_indices = _generate_qkv(
        seq_lens, # [0, 3, 256, 513, 1023, 2048]
        page_size, # 16
        max_kv_len, # 2048
        num_kv_heads, # 8
        num_kv_heads * q_kv_head_ratio, # 64
        head_dim, # 128
        jax.random.key(0),
        dtype,
        are_kv_quantized=are_kv_quantized,
    )
    o = paged_attention.paged_attention(
        q, # [batch_size, num_heads, head_dim]
        k_pages, # [num_kv_heads, total_num_pages, page_size, head_dim]=(8, 768, 16, 128)
        v_pages, # [num_kv_heads, total_num_pages, page_size, head_dim]=(8, 768, 16, 128)
        seq_lens, # i32[batch_size]
        page_indices, # i32[batch_size, pages_per_sequence]=(6, 128)
        pages_per_compute_block=block_size // page_size,
        megacore_mode=megacore_mode,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    k = _reconstruct_kv(page_indices, k_pages)
    v = _reconstruct_kv(page_indices, v_pages)
    import pdb; pdb.set_trace()
    # q: [batch_size, num_heads, head_dim]=(6,64,128)
    # k,v: =[batch_size,num_kv_heads,-1,head_dim]=(6, 8, 2048, 128)
    o_ref = _grouped_query_attention_reference(
        q, k, v, seq_lens, attn_logits_soft_cap)

    if q_kv_head_ratio > 1:
      atol, rtol = 1e-2, 2e-2
    else:
      atol, rtol = 2e-1, 1e-1
    np.testing.assert_allclose(
        o[np.where(seq_lens > 0)].astype(jnp.float32),
        o_ref[np.where(seq_lens > 0)].astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
