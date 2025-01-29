from typing import List, Optional, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
from torch_xla.experimental.pallas_kernels.ragged_paged_attention_kernel import ragged_paged_attention, make_group_metadata
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


# https://github.com/vllm-project/flash-attention/blob/98a4f8df6f5f50413e03f102dc319690300d4aaf/tests/test_vllm_flash_attn.py#L22
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
    cur_q_len = cu_q_lens[i+1] - cu_q_lens[i]
    q = queries[start_idx:start_idx+cur_q_len]  # [cur_q_len, num_q_heads, head_dim]

    cur_kv_len = kv_lens[i]
    num_pages = (cur_kv_len + page_size - 1) // page_size
    page_indices_to_use = page_indices[i, :num_pages]
    k = k_pages[:, page_indices_to_use, :, :]
    k = jnp.permute_dims(k, (1, 2, 0, 3))
    k = jnp.reshape(k, (-1, num_kv_heads, head_dim))
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
        jnp.int32, (cur_q_len, cur_kv_len), 0
    )
    kv_span = jax.lax.broadcasted_iota(jnp.int32, (cur_q_len, cur_kv_len), 1)
    mask = jnp.where(q_span < kv_span, float("-inf"), 0.)
    with jax.numpy_rank_promotion("allow"):
      attn = attn + mask
    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v)  # [cur_q_len, num_q_heads, head_dim]

    outputs.append(out)
    start_idx += cur_q_len

  return jnp.concatenate(outputs, axis=0)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionKernelTest(jtu.JaxTestCase):

  def _verify_ragged_paged_attention_debug(
      self,
      seq_lens,
      num_heads,
      head_dim,
      page_size,
      dtype,
      num_pages,
  ):
    num_seqs = len(seq_lens)
    query_lens = [seq_len[0] for seq_len in seq_lens]
    num_q_tokens = sum(query_lens)
    kv_lens = jnp.array([seq_len[1] for seq_len in seq_lens])
    num_q_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_q_heads % num_kv_heads == 0, "num_q_heads % num_kv_heads !=0."

    prng_key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(prng_key, 4)
    queries = jax.random.normal(k1,
                                (num_q_tokens, num_q_heads, head_dim),
                                dtype=dtype)
    k_pages = jax.random.normal(k2,
                                (num_kv_heads, num_pages, page_size, head_dim),
                                dtype=dtype)
    v_pages = jax.random.normal(k3,
                                (num_kv_heads, num_pages, page_size, head_dim),
                                dtype=dtype)
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
    max_num_pages_per_seq = self._get_closest_power_of_two(max_num_pages_per_seq)
    print(f"xw32 max_kv_len: {max_kv_len}, {max_num_pages_per_seq=}")
    # The assert below mimics the reality that each page get a unique index.
    # But for testing, the assert could be omitted.
    assert max_num_pages_per_seq*num_q_tokens <= num_pages, f"assert failed: max_num_pages_per_seq*num_q_tokens < num_pages. Got {max_num_pages_per_seq*num_q_tokens} and {num_pages}"
    page_indices = jax.random.randint(k4, (num_q_tokens, max_num_pages_per_seq), 0, num_pages, dtype=jnp.int32)
    # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
    q_lens_with_paddings = [0] * num_q_tokens
    for i in range(num_seqs):
      q_lens_with_paddings[i] = query_lens[i]
    cu_q_lens = jnp.cumsum(jnp.array([0]+q_lens_with_paddings))

    actual_output = ragged_paged_attention(
        queries,
        k_pages,
        v_pages,
        kv_lens_np,
        page_indices,
        cu_q_lens,
        num_seqs,
    )
    actual_output = jax.block_until_ready(actual_output)
    print("ragged paged attention finished.")

    expected_output = _ref_ragged_paged_attention(
        queries,
        k_pages,
        v_pages,
        kv_lens_np,
        page_indices,
        cu_q_lens,
        num_seqs,
    )

    self.assertEqual(actual_output.shape, expected_output.shape)
    self.assertEqual(actual_output.dtype, expected_output.dtype)

    print(f'xw32 {expected_output[:192]=}')
    print(f'xw32 {actual_output[:192]=}')

    print(f'Output max diff: {jnp.max(jnp.abs(expected_output - actual_output))}')
    print(f'Output mean diff: {jnp.mean(jnp.abs(expected_output - actual_output))}')
    if dtype == jnp.float32:
      atol = 2e-2
      rtol = 1e-2
    elif dtype == jnp.bfloat16:
      atol = 6e-1
      rtol = 1e-1
    else:
      self.fail(f'Unsupported dtype: {dtype}')
    self.assertTrue(jnp.allclose(actual_output[:128], expected_output[:128], atol=atol, rtol=rtol))
    self.assertTrue(jnp.allclose(actual_output[128:192], expected_output[128:192], atol=atol, rtol=rtol))
    self.assertTrue(jnp.allclose(actual_output[192:256], expected_output[192:256], atol=atol, rtol=rtol))
    self.assertTrue(jnp.allclose(actual_output, expected_output, atol=atol, rtol=rtol))

  def _verify_ragged_paged_attention(
      self,
      seq_lens,
      num_heads,
      head_dim,
      page_size,
      dtype,
      num_pages,
  ):
    num_seqs = len(seq_lens)
    query_lens = [seq_len[0] for seq_len in seq_lens]
    num_q_tokens = sum(query_lens)
    kv_lens = jnp.array([seq_len[1] for seq_len in seq_lens])
    num_q_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_q_heads % num_kv_heads == 0, "num_q_heads % num_kv_heads !=0."

    prng_key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(prng_key, 4)
    queries = jax.random.normal(k1,
                                (num_q_tokens, num_q_heads, head_dim),
                                dtype=dtype)
    k_pages = jax.random.normal(k2,
                                (num_kv_heads, num_pages, page_size, head_dim),
                                dtype=dtype)
    v_pages = jax.random.normal(k3,
                                (num_kv_heads, num_pages, page_size, head_dim),
                                dtype=dtype)
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
    max_num_pages_per_seq = self._get_closest_power_of_two(max_num_pages_per_seq)
    print(f"xw32 max_kv_len: {max_kv_len}, {max_num_pages_per_seq=}")
    # The assert below mimics the reality that each page get a unique index.
    # But for testing, the assert could be omitted.
    assert max_num_pages_per_seq*num_q_tokens <= num_pages, f"assert failed: max_num_pages_per_seq*num_q_tokens < num_pages. Got {max_num_pages_per_seq*num_q_tokens} and {num_pages}"
    page_indices = jax.random.randint(k4, (num_q_tokens, max_num_pages_per_seq), 0, num_pages, dtype=jnp.int32)
    # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
    q_lens_with_paddings = [0] * num_q_tokens
    for i in range(num_seqs):
      q_lens_with_paddings[i] = query_lens[i]
    cu_q_lens = jnp.cumsum(jnp.array([0]+q_lens_with_paddings))

    actual_output = ragged_paged_attention(
        queries,
        k_pages,
        v_pages,
        kv_lens_np,
        page_indices,
        cu_q_lens,
        num_seqs,
    )
    actual_output = jax.block_until_ready(actual_output)
    print("ragged paged attention finished.")

    expected_output = _ref_ragged_paged_attention(
        queries,
        k_pages,
        v_pages,
        kv_lens_np,
        page_indices,
        cu_q_lens,
        num_seqs,
    )

    self.assertEqual(actual_output.shape, expected_output.shape)
    self.assertEqual(actual_output.dtype, expected_output.dtype)

    print(f'Output max diff: {jnp.max(jnp.abs(expected_output - actual_output))}')
    print(f'Output mean diff: {jnp.mean(jnp.abs(expected_output - actual_output))}')
    if dtype == jnp.float32:
      atol = 2e-2
      rtol = 1e-2
    elif dtype == jnp.bfloat16:
      atol = 6e-1
      rtol = 1e-1
    else:
      self.fail(f'Unsupported dtype: {dtype}')
    self.assertTrue(jnp.allclose(actual_output, expected_output, atol=atol, rtol=rtol))

  def _get_closest_power_of_two(self, x):
    if x <= 0:
      raise ValueError(f"x must be positive. Got {x}")
    return 2 ** int(np.ceil(np.log2(x)))

  def test_paged_attention_min_two_kv_block_per_sequence(
      self,
  ):
    # assuming q_blk_size=128, page_size=16, num_kv_pages_per_compute_block=16
    # One of the constraints of the kernel is that q.shape[0]%q_blk_size==0 as in _calculate_num_tiles.
    # If we cannot get the assumption, we can pad the matrix q in the kernel.
    seq_lens = [(192, 328), (128, 180), (64, 255)]  # [(q_len, kv_len),...]
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 65536

    self._verify_ragged_paged_attention_debug(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  def test_paged_attention_basic(
      self,
  ):
    # assuming q_blk_size=128
    seq_lens = [(192, 1328), (128, 180), (64, 463)]  # [(q_len, kv_len),...]
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 65536

    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )


  def test_paged_attention_basic_with_one_token_per_sequence(
      self,
  ):
    # assuming q_blk_size=128
    seq_lens = [(1, 127), (120, 1328), (1, 64), (1, 64), (1, 64), (1, 64), (256, 256), (131, 463)]  # [(q_len, kv_len),...]
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 65536

    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  def test_paged_attention_extreme_all_tokens_belong_to_one_sequence(
      self,
  ):
    # assuming q_blk_size=128
    seq_lens = [(512, 1328)]  # [(q_len, kv_len),...]
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 65536

    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  def test_paged_attention_extreme_one_tokens_per_sequence(
      self,
  ):
    # assuming q_blk_size=128
    seq_lens = []  # [(q_len, kv_len),...]
    num_seqs = 512
    for i in range(num_seqs):
      seq_lens.append((1, i))
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 65536

    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  def test_make_sequence_metadata(
      self,
  ):
    cu_q_lens = jnp.array([0, 192, 448, 512] + [512]*(512-4))
    num_q_tokens = 512
    num_queries_per_compute_block = 128
    start_group = jnp.array([0])
    num_seqs = 3
    metadata, num_logical_q_tiles  = make_group_metadata(
        cu_q_lens=cu_q_lens,
        m=num_q_tokens,
        tm=num_queries_per_compute_block,
        start_group=start_group,
        num_seqs=num_seqs
    )
    seq_ids, physical_q_tile_ids = metadata
    # print(f"xw32 metadata.physical_q_tile_ids: {metadata.physical_q_tile_ids}")
    # print(f"xw32 metadata.seq_ids: {metadata.seq_ids}")
    self.assertEqual(num_logical_q_tiles, 6)
    self.assertTrue(jnp.array_equal(seq_ids, [0, 0, 1, 1, 1, 2]))
    self.assertTrue(jnp.array_equal(physical_q_tile_ids, [0, 1, 1, 2, 3, 3]))
    # print('xw32======================')
    # q_lens = jnp.array([192, 256, 64] + [0]*(512-3))
    # metadata = ragged_paged_attention_kernel.original_make_group_metadata(
    #     group_sizes=q_lens,
    #     m=num_q_tokens,
    #     tm=num_queries_per_compute_block,
    #     start_group=start_group,
    #     num_nonzero_groups=num_seqs,
    #     visit_empty_groups=False,
    # )
    # print(f"xw32 {metadata=}")
    # self.assertEqual(metadata.num_logical_q_tiles, 6)
    # print(f"xw32 metadata.seq_ids: {metadata.seq_ids}")
    # print(f"xw32 metadata.physical_q_tile_ids: {metadata.physical_q_tile_ids}")
    # print(f"xw32 metadata.seq_ids[:metadata.num_logical_q_tiles]: {metadata.seq_ids[:metadata.num_logical_q_tiles]}")
    # self.assertTrue(jnp.array_equal(metadata.seq_ids[:metadata.num_logical_q_tiles], [0, 0, 1, 1, 1, 2]))
    # self.assertTrue(jnp.array_equal(metadata.physical_q_tile_ids[:metadata.num_logical_q_tiles], [0, 1, 1, 2, 3, 3]))



if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
