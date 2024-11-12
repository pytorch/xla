from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from torch_xla.experimental.pallas_kernels.multi_queries_paged_attention_kernel import paged_attention
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
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype)

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(
      k4, (batch_size, query_len, num_q_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices


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


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

  #   def test_paged_attention(
  #       self,
  #   ):
  #     dtype = jnp.bfloat16
  #     page_size=16
  #     num_kv_heads = 8
  #     q_kv_head_ratio = 4
  #     head_dim = 256
  #     num_queries_per_compute_block = 32
  #     block_kv_size = 256

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      page_size=(16, 32, 64),
      num_kv_heads=(1, 8),
      q_kv_head_ratio=(1, 4, 8),
      head_dim=(128, 256),
      num_queries_per_compute_block=(16, 32),
      block_kv_size=(128, 256),
  )
  def test_paged_attention_without_query_padding(
      self,
      dtype,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      num_queries_per_compute_block,
      block_kv_size,
  ):

    max_kv_len = 2048
    query_len = 64
    batch_size = 3
    kv_seq_lens = jax.random.randint(
        jax.random.key(0), (batch_size,), query_len, max_kv_len)

    assert query_len <= max_kv_len
    for cur_kv_seq in kv_seq_lens:
      assert query_len <= cur_kv_seq, f'{query_len} should be less than or equal to the kv_len {cur_kv_seq} in the current sequence.'
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

    print(f'Running paged_attention with {query_len=}')
    num_kv_pages_per_compute_block = block_kv_size // page_size
    effective_q_lens = jnp.full_like(kv_seq_lens, query_len)
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
    # actual_output = jax.block_until_ready(actual_output)

    # Run the ref impl.
    expected_output = _ref_jax_extended_paged_attention(
        q,
        k_pages,
        v_pages,
        kv_seq_lens,
        page_indices,
        effective_q_lens,
    )

    self.assertEqual(actual_output.shape, expected_output.shape)

    if dtype == jnp.float32:
      atol = 1e-2
      rtol = 1e-2
    elif dtype == jnp.bfloat16:
      atol = 6e-1
      rtol = 1e-1
    else:
      self.fail(f'Unsupported dtype: {dtype}')
    self.assertTrue(
        jnp.allclose(expected_output, actual_output, atol=atol, rtol=rtol))

  # def test_paged_attention_query_len_longer_than_kv_seq_len(
  #     self,
  # ):
  #   dtype = jnp.float32
  #   page_size=16
  #   num_kv_heads = 8
  #   q_kv_head_ratio = 4
  #   head_dim = 256
  #   num_queries_per_compute_block = 32
  #   block_kv_size = 256
  # In practice, vLLM would pad the query so that the query seq len will be longer than the kv seq len. query seq len may be padded but not for kv seq len.
  # When this happens, we need an additional parameter `effective_q_lens` to the paged_attention to set the causal mask right.
  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      page_size=(16, 32, 64),
      num_kv_heads=(1, 8),
      q_kv_head_ratio=(1, 4, 8),
      head_dim=(128, 256),
      num_queries_per_compute_block=(16, 32),
      block_kv_size=(128, 256),
  )
  def test_paged_attention_with_query_padding(
      self,
      dtype,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      num_queries_per_compute_block,
      block_kv_size,
  ):

    max_kv_len = 2048
    # Set query_len>kv_seq_lens
    query_len = max_kv_len
    batch_size = 3
    kv_seq_lens = jax.random.randint(
        jax.random.key(0), (batch_size,), 0, max_kv_len)
    effective_q_lens = jax.random.randint(
        jax.random.key(0), (batch_size,), 0, kv_seq_lens)
    for cur_effec_q_len, cur_kv_seq_len in zip(effective_q_lens, kv_seq_lens):
      assert cur_effec_q_len <= cur_kv_seq_len, f'The effective query len {cur_effec_q_len} should be less than or equal to the kv_len {cur_kv_seq_len} in the current sequence.'

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

    print(
        f'Running paged_attention with {query_len=}, {kv_seq_lens=}, {effective_q_lens=}'
    )
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
    # actual_output = jax.block_until_ready(actual_output)

    # Run the ref impl.
    expected_output = _ref_jax_extended_paged_attention(
        q,
        k_pages,
        v_pages,
        kv_seq_lens,
        page_indices,
        effective_q_lens,
    )

    self.assertEqual(actual_output.shape, expected_output.shape)

    if dtype == jnp.float32:
      atol = 2e-2
      rtol = 1e-2
    elif dtype == jnp.bfloat16:
      atol = 6e-1
      rtol = 1e-1
    else:
      self.fail(f'Unsupported dtype: {dtype}')
    for b in range(batch_size):
      # N.B. For the output ([batch_size, query_len, num_q_heads, head_dim]) at query_len dim, all the value after the effective_q_len will be thrown away due to we padding the query seq len. The values after the effective_q_len may differ between the kernel and the ref impl because of the causal mask.
      effective_q_len = effective_q_lens[b]
      self.assertTrue(
          jnp.allclose(
              expected_output[b, :effective_q_len],
              actual_output[b, :effective_q_len],
              atol=atol,
              rtol=rtol))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
