from typing import List, Optional, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
from torch_xla.experimental.pallas_kernels.ragged_paged_attention_kernel import ragged_paged_attention, make_sequence_metadata, DEFAULT_MASK_VALUE
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

ATOL_FP32 = 2e-1


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
            3))  #   [page_indices_to_use, page_size, num_kv_heads, head_dim]
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


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionKernelTest(jtu.JaxTestCase):

  def _verify_ragged_paged_attention(
      self,
      seq_lens,
      num_heads,
      head_dim,
      page_size,
      dtype,
      num_pages,
      num_kv_pages_per_block=128,
      num_queries_per_block=128,
  ):
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
    max_num_pages_per_seq = self._round_up_closest_multiple_of(
        max_num_pages_per_seq, num_kv_pages_per_block)
    # The assert below mimics the reality that each page get a unique index.
    # But for testing, the assert could be omitted.
    # assert max_num_pages_per_seq*num_q_tokens <= num_pages, f"assert failed: max_num_pages_per_seq*num_q_tokens < num_pages. Got {max_num_pages_per_seq*num_q_tokens} and {num_pages}"
    page_indices = jax.random.randint(
        k4, (num_q_tokens, max_num_pages_per_seq),
        0,
        num_pages,
        dtype=jnp.int32)

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
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )
    err.throw()  # noop if there is not err.
    actual_output = jax.block_until_ready(actual_output)

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

    print(
        f'Output max diff: {jnp.max(jnp.abs(expected_output - actual_output))}')
    print(
        f'Output mean diff: {jnp.mean(jnp.abs(expected_output - actual_output))}'
    )
    if dtype == jnp.float32:
      atol = 2e-1
      rtol = 1e-2
    elif dtype == jnp.bfloat16:
      atol = 6e-1
      rtol = 1e-1
    else:
      self.fail(f'Unsupported dtype: {dtype}')
    self.assertTrue(
        jnp.allclose(actual_output, expected_output, atol=atol, rtol=rtol))

  def _round_up_closest_multiple_of(self, x, base):
    return (x + base - 1) // base * base

  def _get_closest_power_of_two(self, x):
    if x <= 0:
      raise ValueError(f"x must be positive. Got {x}")
    return 2**int(np.ceil(np.log2(x)))

  def test_paged_attention_basic(self,):
    # Same setup as in the design doc.
    # assuming q_blk_size=128, page_size=16, num_kv_pages_per_compute_block=16
    # Note one of the constraints of the kernel is that q.shape[0]%q_blk_size==0 as in _calculate_num_tiles.
    seq_lens = [(192, 328), (128, 180), (64, 255)]  # [(q_len, kv_len),...]
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

  @parameterized.product(
      seq_lens=[[(1, 1328), (5, 18), (506, 563)]],
      num_heads=[(4, 4), (8, 2), (16, 2)],
      head_dim=[128, 256],
      dtype=(jnp.float32, jnp.bfloat16),
      page_size=[16, 32],
      num_pages=[32768, 2048],
  )
  def test_paged_attention_varlen_comprehensive(
      self,
      seq_lens: List[Tuple[int, int]],
      num_heads: Tuple[int, int],
      head_dim: int,
      dtype,
      page_size: int,
      num_pages: int,
  ):
    if jtu.is_device_tpu(version=4) and head_dim == 256 and page_size == 32:
      self.skipTest(
          "TPU v4 has small VMEM. It will run into VMEM OOM. Skip the test.")
    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
        num_queries_per_block=64,
    )

  def test_paged_attention_mix_prefill_and_decode1(self,):
    # assuming q_blk_size=128
    seq_lens = [
        (1, 1328),
        (5, 18),
        (1, 129),
        (120, 229),
        (1, 122),  # first physical q block
        (1, 64),
        (32, 100),
        (250, 463),
        (1, 18),
        (1, 17),
        (99, 123)
    ]  # last 3 physical q blocks [(q_len, kv_len),...]
    num_heads = (4, 4)
    head_dim = 128
    dtype = jnp.float32
    page_size = 16
    num_pages = 32768

    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
    )

  def test_paged_attention_mix_prefill_and_decode2(self,):
    # assuming q_blk_size=128
    seq_lens = [(1, 127), (120, 1328), (1, 64), (1, 64), (1, 64), (1, 64),
                (256, 256), (131, 463)]  # [(q_len, kv_len),...]
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

  def test_paged_attention_extreme_all_tokens_belong_to_one_sequence(self,):
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

  def test_paged_attention_extreme_one_tokens_per_sequence_min(self,):
    seq_lens = []  # [(q_len, kv_len),...]
    num_seqs = 64
    num_queries_per_block = 16
    for i in range(num_seqs):
      seq_lens.append((1, 256 + i))
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 1024

    self._verify_ragged_paged_attention(
        seq_lens,
        num_heads,
        head_dim,
        page_size,
        dtype,
        num_pages,
        num_queries_per_block=num_queries_per_block,
    )

  def test_paged_attention_q_len_should_be_no_longer_than_kv_len(self,):
    # assuming q_blk_size=128
    # Here the q_len(1 or 511) is set up to be longer than the corresponding kv_len (0 or 256).
    seq_lens = [(1, 0), (511, 256)]  # [(q_len, kv_len),...]
    num_heads = (1, 1)
    head_dim = 128
    page_size = 16
    dtype = jnp.float32
    num_pages = 65536

    num_seqs = len(seq_lens)
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
    num_kv_pages_per_block = 128
    max_num_pages_per_seq = self._round_up_closest_multiple_of(
        max_num_pages_per_seq, num_kv_pages_per_block)
    # The assert below mimics the reality that each page get a unique index.
    # But for testing, the assert could be omitted.
    assert max_num_pages_per_seq * num_q_tokens <= num_pages, f"assert failed: max_num_pages_per_seq*num_q_tokens < num_pages. Got {max_num_pages_per_seq*num_q_tokens} and {num_pages}"
    page_indices = jax.random.randint(
        k4, (num_q_tokens, max_num_pages_per_seq),
        0,
        num_pages,
        dtype=jnp.int32)

    # Create a cu_q_lens: jax.Array,		# i32[num_tokens + 1]
    q_lens_with_paddings = [0] * num_q_tokens
    for i in range(num_seqs):
      q_lens_with_paddings[i] = query_lens[i]
    cu_q_lens = jnp.cumsum(jnp.array([0] + q_lens_with_paddings))

    with self.assertRaisesRegex(
        ValueError, "cur_q_len must be less or equal to cur_kv_len"):
      err, _ = ragged_paged_attention(
          queries,
          k_pages,
          v_pages,
          kv_lens_np,
          page_indices,
          cu_q_lens,
          num_seqs,
          num_kv_pages_per_block=num_kv_pages_per_block,
      )
      err.throw()

  def test_paged_attention_extreme_one_tokens_per_sequence_large(self,):
    # assuming q_blk_size=128
    seq_lens = []  # [(q_len, kv_len),...]
    num_seqs = 512
    for i in range(num_seqs):
      seq_lens.append((1, 128 + i))
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

  def test_make_sequence_metadata(self,):
    cu_q_lens = jnp.array([0, 192, 448, 512] + [512] * (512 - 4))
    num_q_tokens = 512
    num_queries_per_compute_block = 128
    start_group = jnp.array([0])
    num_seqs = 3
    metadata, num_logical_q_tiles = make_sequence_metadata(
        cu_q_lens=cu_q_lens,
        m=num_q_tokens,
        tm=num_queries_per_compute_block,
        start_sequence=start_group,
        num_sequences=num_seqs)
    seq_ids, physical_q_tile_ids = metadata
    self.assertEqual(num_logical_q_tiles, 6)
    self.assertTrue(jnp.array_equal(seq_ids, [0, 0, 1, 1, 1, 2]))
    self.assertTrue(jnp.array_equal(physical_q_tile_ids, [0, 1, 1, 2, 3, 3]))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
