from collections.abc import Sequence
from collections import namedtuple
import functools
from typing import Any, Literal, Optional, cast

import jax
from jax import lax
from jax.experimental import checkify
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [num_kv_heads, total_num_pages, page_size, head_dim]
      scales_pages_hbm_ref,
      vmem_buffer,  # [pages_per_compute_block, page_size, head_dim]
      scales_vmem_buffer,
      sem,
      page_indices,  # [num_kv_pages_per_block]
      num_pages_to_load,
      kv_head_index,
  ):
    # Original k_pages has shape [num_kv_heads, total_num_pages, page_size, head_dim]
    assert page_indices.shape[0] == num_pages_to_load
    self._vmem_buffer = vmem_buffer
    self._scales_vmem_buffer = scales_vmem_buffer
    self._num_pages_to_load = num_pages_to_load
    if kv_head_index is not None:
      self._pages_hbm_ref = pages_hbm_ref.at[kv_head_index]
      if scales_pages_hbm_ref is not None:
        self._scales_pages_hbm_ref = scales_pages_hbm_ref.at[kv_head_index]
      else:
        self._scales_pages_hbm_ref = None
    else:
      self._pages_hbm_ref = pages_hbm_ref
      self._scales_pages_hbm_ref = scales_pages_hbm_ref
    self._sem = sem
    self._page_indices = page_indices
    self._async_copies = [
        self._make_async_copy(i) for i in range(self._num_pages_to_load)
    ]
    if (self._scales_pages_hbm_ref is not None and
        self._scales_vmem_buffer is not None):
      self._async_copies += [
          self._make_scales_async_copy(i)
          for i in range(self._num_pages_to_load)
      ]

  def _make_async_copy(self, i):
    page_index = self._page_indices[i]
    return pltpu.make_async_copy(self._pages_hbm_ref.at[page_index],
                                 self._vmem_buffer.at[i], self._sem)

  def _make_scales_async_copy(self, i):
    page_index = self._page_indices[i]
    return pltpu.make_async_copy(
        self._scales_pages_hbm_ref.at[page_index],  # pytype: disable=attribute-error
        self._scales_vmem_buffer.at[i],  # pytype: disable=attribute-error
        self._sem,
    )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def _maybe_dequantize(self, x, x_scale, dtype=jnp.bfloat16):
    if x_scale is None:
      return x.astype(dtype)
    return quantization_utils.from_int8(x, x_scale, dtype=dtype)

  def wait_and_get_loaded(self) -> jax.Array:
    """Wait async copies and gets the loaded buffer as a jax.Array."""
    # Return value shape is [pages_per_compute_block*page_size, head_dim]
    for async_copy in self._async_copies:
      async_copy.wait()
    head_dim = self._vmem_buffer.shape[-1]
    jax_array = self._vmem_buffer[...].astype(jnp.float32)
    if self._scales_vmem_buffer is not None:
      scales_jax_array = self._scales_vmem_buffer[...].astype(jnp.float32)
    else:
      scales_jax_array = None
    jax_array = self._maybe_dequantize(jax_array, scales_jax_array)
    return jax_array.reshape(-1, head_dim)


def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
  return tiles


# https://github.com/jax-ml/jax/blob/9fb29766a2130e74a85cba30420cf777d185ea5a/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L79
def make_sequence_metadata(
    *,
    cu_q_lens: jnp.ndarray,
    m: int,
    tm: int,
    start_sequence: jnp.ndarray,
    num_sequences: int,
):
  """Create the metadata needed for ragged paged attention computation.

  Args:
    cu_q_lens: : A 1d, jnp.ndarray with shape [num_seqs+1] and jnp.int32 dtype.
      The cumulative query lengths.
    m: The number of query tokens.
    tm: The m-dimension tile size being used.
    start_sequence: The sequence in cu_q_lens to start computing from. This is useful for when num_seqs is sharded.
    num_sequences: The number of sequences to compute on.

  Returns:
    tuple of:
      seq_ids: A 1d, jnp.ndarray with shape [m_tiles + num_seqs] and
        jnp.int32 dtype. seq_ids[i] indicates which sequence the grid index (num_logical_tiles_q) will work on.
      physical_q_tile_ids: A 1d, jnp.ndarray with shape [m_tiles + num_seqs] and
        jnp.int32. physical_q_tile_ids[i] indicates which query-dim physical tile the grid index (num_logical_tiles_q) will work on.

    num_logical_q_tiles: The number of query-dim logical tiles to execute.
  """
  end_sequence = start_sequence + num_sequences - 1

  # We need the offset of each sequence from input, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # sequence_offsets.shape = [num_sequences + 1]
  # sequence_offsets[0] = 0
  # sequence_offsets[num_sequences] = m
  #
  # The row at which sequence 'i' starts is sequence_offsets[i].
  sequence_ends = cu_q_lens[1:]
  sequence_offsets = cu_q_lens

  # Assign a sequence id to each grid index. The grid index refers to the logical q tile index.
  #
  # If a sequence starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each sequence by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.

  # (1) Round the sequence_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change sequence_offsets[num_sequences], which is m
  # (because we enforce m is divisible by tm).
  rounded_sequence_ends = ((sequence_ends + tm - 1) // tm * tm).astype(
      jnp.int32)

  # (2) Round the sequence_starts down to the nearest multiple of 'tm'.
  sequence_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), sequence_ends[:-1]])
  rounded_sequence_starts = sequence_starts // tm * tm

  # (3) Calculate the number of rows in each sequence.
  rounded_sequence_sizes = rounded_sequence_ends - rounded_sequence_starts

  # (4) Convert the sequence sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by sequence 'i' if the first row of the tile
  # belongs to sequence 'i'. In addition to owned tiles, each sequence can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' sequence never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no sequence has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every sequence has a partial except the 0-th sequence, the total
  # number of tiles is equal to 'm // tm + num_sequences - 1'. Thus we know that
  #
  # tiles_m <= sequence_tiles.sum() <= tiles_m + num_sequences - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All sequence sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  sequence_tiles = rounded_sequence_sizes // tm

  # Create the sequence ids for each grid index based on the tile counts for each
  # sequence.
  #
  # NOTE: This repeat(...) will pad sequence_ids with the final sequence id if
  # sequence_tiles.sum() < tiles_m + num_sequences - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)
  sequence_ids = jnp.repeat(
      jnp.arange(num_sequences, dtype=jnp.int32),
      sequence_tiles[:num_sequences],
      total_repeat_length=tiles_m + num_sequences - 1,
  )

  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the sequence that owns the tile.
  # The remaining possible visits occur when a sequence starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each sequence starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the sequence that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram.
  #
  partial_tile_mask = ((sequence_offsets[:-1] % tm) == 0)

  partial_tile_ids = jnp.where(partial_tile_mask, tiles_m,
                               sequence_offsets[:-1] // tm)

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] +
      1)

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = jnp.repeat(
      jnp.arange(tiles_m, dtype=jnp.int32),
      tile_visits.astype(jnp.int32),
      total_repeat_length=tiles_m + num_sequences - 1,
  )

  # Account for sharding.
  #
  # Find the start of the sequences owned by our shard and shift the sequence_ids and
  # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
  #
  first_tile_in_shard = (sequence_ids < start_sequence).sum()
  sequence_ids = jnp.roll(sequence_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

  # Calculate the number of tiles we need to compute for our shard.
  #
  # Remove tile visits that belong to a sequence not in our shard.
  iota = jnp.arange(num_sequences, dtype=jnp.int32)
  active_sequence_mask = jnp.logical_and(iota <= end_sequence, iota
                                         >= start_sequence)
  sequence_tiles = jnp.where(active_sequence_mask,
                             sequence_tiles[:num_sequences], 0)
  num_tiles = sequence_tiles.sum()
  return (sequence_ids, m_tile_ids
         ), num_tiles  # (seq_ids, physical_q_tile_ids), num_logical_q_tiles


def check_kernel_input(q, k_pages, v_pages, kv_lens, page_indices, cu_q_lens,
                       num_seqs, num_kv_pages_per_block):
  num_q_heads, num_tokens, head_dim = q.shape
  num_kv_heads, _, _, head_dim_k = k_pages.shape
  _, pages_per_sequence = page_indices.shape
  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"  # pytype: disable=attribute-error
    )
  if head_dim_k != head_dim:
    raise ValueError("head_dim of Q must be the same as that of K/V. Got"
                     f" {head_dim} and {head_dim_k}.")
  if kv_lens.shape[0] != num_tokens:
    raise ValueError("kv_lens.shape[0] must be the same as num_tokens. Got"
                     f" {kv_lens.shape[0]} and {num_tokens}")
  if page_indices.shape[0] != num_tokens:
    raise ValueError("page_indices.shape[0] must be the same as num_tokens. Got"
                     f" {page_indices.shape[0]} and {num_tokens}")
  if cu_q_lens.shape[0] != num_tokens + 1:
    raise ValueError(
        "cu_q_lens.shape[0] must be the same as num_tokens + 1. Got"
        f" {cu_q_lens.shape[0]} and {num_tokens + 1}")
  for i in range(num_seqs):
    cur_q_len = cu_q_lens[i + 1] - cu_q_lens[i]
    cur_kv_len = kv_lens[i]
    checkify.check(
        cur_q_len <= cur_kv_len,
        "cur_q_len must be less or equal to cur_kv_len. Got {} and {}",
        cur_q_len, cur_kv_len)
  if num_seqs > num_tokens:
    raise ValueError(
        f"num_seqs must be less or equal to num_tokens. Got {num_seqs} and {num_tokens}"
    )
  if kv_lens.dtype != jnp.int32 or page_indices.dtype != jnp.int32 or cu_q_lens.dtype != jnp.int32:
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {kv_lens.dtype=}, "
        f"{page_indices.dtype=}, {cu_q_lens.dtype=}")
  if num_kv_pages_per_block > pages_per_sequence:
    raise ValueError(
        f"{num_kv_pages_per_block=} should be smaller or equal to {pages_per_sequence=}"
    )
  # The below constraint "num_kv_pages_per_block % PALLAS_LAST_DIM_MIN_SIZE == 0" comes from when we chunk the page_indices and load it into SMEM. Pallas requires the last dim to be a multiple of 128.
  PALLAS_LAST_DIM_MIN_SIZE = 128
  if num_kv_pages_per_block % PALLAS_LAST_DIM_MIN_SIZE != 0:
    raise ValueError(
        f"{num_kv_pages_per_block=} should be a multiple of {PALLAS_LAST_DIM_MIN_SIZE=}"
    )
  if pages_per_sequence % num_kv_pages_per_block != 0:
    raise ValueError(
        "pages_per_sequence must be divisible by num_kv_pages_per_block. Got"
        f" {pages_per_sequence=} and {num_kv_pages_per_block=}.")
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_q_heads} and {num_kv_heads}.")


# https://github.com/jax-ml/jax/blob/e3b3b913f7bcec3767e1442ace08999413f8703d/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L269C1-L283C64
def _get_store_mask(
    *,
    logical_q_blk_idx: jnp.ndarray,
    sequence_offsets: jnp.ndarray,
    sequence_ids: jnp.ndarray,
    physical_q_tile_ids: jnp.ndarray,
    tq: int,
    tk: int,
) -> jnp.ndarray:
  """Mask for rows that belong to the current sequence in the current physical q tile."""
  sequence_id = sequence_ids[logical_q_blk_idx]
  sequence_start = sequence_offsets[sequence_id]
  sequence_end = sequence_offsets[sequence_id + 1]
  physical_q_tile_id = physical_q_tile_ids[logical_q_blk_idx] * tq
  iota = jax.lax.broadcasted_iota(jnp.int32, (tq, tk), 0) + physical_q_tile_id
  return jnp.logical_and(iota >= sequence_start, iota < sequence_end)


def _flash_attention(
    q_head_idx_per_kv,  # scalar, ranges from 0 to num_query_heads_per_kv_head
    sequence_metadata_ref,  # Tuple (seq_ids, physical_q_tile_ids)
    effective_kv_lens_ref,  # [num_tokens]
    effective_cu_q_lens_ref,  # [num_tokens + 1]
    # kernel inputs
    q_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    k,  # [kv_blk_size, head_dim]
    v,  # [kv_blk_size, head_dim]
    # outputs
    o_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    l_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    m_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    # scratch space
    l_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    m_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    acc_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    *,
    num_tokens: int,
    num_seqs: int,
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    mask_value: float,
    page_size: int,
    head_dim: int,
    num_q_heads_per_kv_head: int,
    sm_scale: float,
):
  assert q_ref.shape == (num_q_heads_per_kv_head, num_queries_per_block,
                         head_dim)
  kv_blk_size = page_size * num_kv_pages_per_block
  assert k.shape == (kv_blk_size, head_dim)
  assert v.shape == (kv_blk_size, head_dim)

  kv_head_idx, logical_q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
  )
  seq_ids, physical_q_tile_ids = sequence_metadata_ref

  # If the q-dim physical tile is changed (meaning it is a new physical q-dim tile that has not visited before), initialize the acc_scratch_ref, m_scratch_ref, and l_scratch_ref to run the flash attention v2 algorithm.
  prev_logical_q_blk_idx = jnp.where(logical_q_blk_idx > 0,
                                     logical_q_blk_idx - 1, 0)
  is_first_processed_logical_q_blk = logical_q_blk_idx == 0
  physical_q_blk_changed = (
      physical_q_tile_ids[logical_q_blk_idx]
      != physical_q_tile_ids[prev_logical_q_blk_idx])
  first_time_seeing_physical_q_blk = jnp.logical_or(
      is_first_processed_logical_q_blk, physical_q_blk_changed)
  is_first_kv_blk = (kv_blk_idx == 0)
  should_init_scratch_ref = jnp.logical_and(is_first_kv_blk,
                                            first_time_seeing_physical_q_blk)

  @pl.when(should_init_scratch_ref)
  def init_scratch_ref():  # pylint: disable=unused-variable
    l_scratch_ref[q_head_idx_per_kv] = jnp.zeros(
        l_scratch_ref[q_head_idx_per_kv].shape, jnp.float32)
    m_scratch_ref[q_head_idx_per_kv] = jnp.full(
        m_scratch_ref[q_head_idx_per_kv].shape, -jnp.inf, jnp.float32)
    acc_scratch_ref[q_head_idx_per_kv] = jnp.zeros(
        acc_scratch_ref[q_head_idx_per_kv].shape, jnp.float32)

  m_prev = m_scratch_ref[
      q_head_idx_per_kv]  # [num_queries_per_block, MIN_BLOCK_SIZE]
  l_prev = l_scratch_ref[
      q_head_idx_per_kv]  # [num_queries_per_block, MIN_BLOCK_SIZE]

  # Load the whole q_block that belongs to the current physical q_blk and compute the attention. When we write, we only write the part that belongs to the current sequence.
  # Cannot just load only the part of q_block that belongs to the current sequence, because it results in dynamic shapes and then fails the JIT compilation.
  # Note, q_ref.shape=[num_q_heads_per_kv_head, num_queries_per_block, head_dim]
  q = q_ref[q_head_idx_per_kv, :, :].astype(jnp.float32)  # [block_q, head_dim]
  assert q.shape == (num_queries_per_block, head_dim)
  s = jnp.einsum(
      'qd,td->qt', q, k,
      preferred_element_type=jnp.float32)  # [block_q, block_k]
  assert s.shape == (num_queries_per_block, kv_blk_size)
  s = s * sm_scale

  # Modify the mask accordingly: first form the mask. Then move the mask up/down to the right place.
  cur_seq_idx = seq_ids[logical_q_blk_idx]
  cur_seq_start = effective_cu_q_lens_ref[cur_seq_idx]
  cur_seq_end = effective_cu_q_lens_ref[cur_seq_idx + 1]
  physical_q_blk_idx = physical_q_tile_ids[logical_q_blk_idx]
  q_index = physical_q_blk_idx * num_queries_per_block - cur_seq_start
  kv_index = kv_blk_idx * kv_blk_size
  effective_kv_len = effective_kv_lens_ref[cur_seq_idx]
  effective_q_len = cur_seq_end - cur_seq_start
  row_ids = (effective_kv_len -
             effective_q_len) + q_index + jax.lax.broadcasted_iota(
                 jnp.int32, (num_queries_per_block, kv_blk_size), 0)
  col_ids = kv_index + jax.lax.broadcasted_iota(
      jnp.int32, (num_queries_per_block, kv_blk_size), 1)
  causal_mask = jnp.where(row_ids < col_ids, mask_value, 0.)
  assert causal_mask.shape == (num_queries_per_block, kv_blk_size)

  s = s + causal_mask  # [block_q, block_k]

  m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
  m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

  block_k_repeats, rem = divmod(kv_blk_size, MIN_BLOCK_SIZE)
  if rem:
    raise NotImplementedError(
        f"{kv_blk_size=} should be a multiple of {MIN_BLOCK_SIZE}")
  p = jnp.exp(
      s - pltpu.repeat(m_next, block_k_repeats, 1))  # Shape [block_q, block_k]

  alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128]

  l_corr = alpha * l_prev  # Shape [block_q, 128]

  l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

  head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
  l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
  if rem:
    if head_dim_repeats == 0:
      l_broadcast = lambda l: l[:, :head_dim]
    else:
      raise NotImplementedError(
          f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")

  # Need to store these l_next and m_next which will relay to the output.
  # But only update the part that belongs to the current sequence we are working on.
  lm_mask = _get_store_mask(
      logical_q_blk_idx=logical_q_blk_idx,
      sequence_offsets=effective_cu_q_lens_ref,
      sequence_ids=seq_ids,
      physical_q_tile_ids=physical_q_tile_ids,
      tq=num_queries_per_block,
      tk=MIN_BLOCK_SIZE,
  )
  # Either jax.lax.select or jnp.where works here.
  pl.store(
      l_scratch_ref,
      (pl.ds(q_head_idx_per_kv, 1), slice(None), slice(None)),
      l_next.reshape(1, *l_next.shape),  # no-op here.
      mask=lm_mask.reshape(1, *lm_mask.shape),
  )
  pl.store(
      m_scratch_ref,
      (pl.ds(q_head_idx_per_kv, 1), slice(None), slice(None)),
      m_next.reshape(1, *m_next.shape),
      mask=lm_mask.reshape(1, *lm_mask.shape),
  )

  l_next_inv_safe = jnp.where(l_next == 0.0, 1.0,
                              1.0 / l_next)  # [block_q, 128]
  temp = acc_scratch_ref[q_head_idx_per_kv] * l_broadcast(
      l_corr * l_next_inv_safe)
  o_curr = jax.lax.dot(
      p.astype(v.dtype), v,
      preferred_element_type=jnp.float32)  # [block_q, 128]
  temp += o_curr * l_broadcast(l_next_inv_safe)
  acc_mask = _get_store_mask(
      logical_q_blk_idx=logical_q_blk_idx,
      sequence_offsets=effective_cu_q_lens_ref,
      sequence_ids=seq_ids,
      physical_q_tile_ids=physical_q_tile_ids,
      tq=num_queries_per_block,
      tk=head_dim,
  )
  pl.store(
      acc_scratch_ref,
      (pl.ds(q_head_idx_per_kv, 1), slice(None), slice(None)),
      temp.reshape(1, *temp.shape),
      mask=acc_mask.reshape(1, *acc_mask.shape),
  )

  # Store the result from VMEM to HBM only when it is the last kv_block and the next q-dim logical tile belongs to a different q-dim physical tile.
  is_last_kv_blk_idx = (
      kv_blk_idx == (pl.cdiv(effective_kv_len, kv_blk_size) - 1))
  num_logical_q_blks = pl.num_programs(
      1)  # grid=(num_kv_heads, num_logical_q_tiles, num_kv_blks)
  next_logical_q_blk_idx = jnp.where(
      logical_q_blk_idx == num_logical_q_blks - 1, logical_q_blk_idx,
      logical_q_blk_idx + 1)
  is_last_logical_q_blk = (logical_q_blk_idx == num_logical_q_blks - 1)
  physical_q_blk_will_change = (
      physical_q_tile_ids[logical_q_blk_idx]
      != physical_q_tile_ids[next_logical_q_blk_idx])
  last_time_seeing_cur_physical_q_blk = jnp.logical_or(
      is_last_logical_q_blk, physical_q_blk_will_change)
  should_store_to_output = jnp.logical_and(is_last_kv_blk_idx,
                                           last_time_seeing_cur_physical_q_blk)

  @pl.when(should_store_to_output)
  def store_to_output():  # pylint: disable=unused-variable
    o_ref[q_head_idx_per_kv] = acc_scratch_ref[q_head_idx_per_kv].astype(
        o_ref.dtype)
    l_ref[q_head_idx_per_kv] = l_scratch_ref[q_head_idx_per_kv].astype(
        l_ref.dtype)
    m_ref[q_head_idx_per_kv] = m_scratch_ref[q_head_idx_per_kv].astype(
        m_ref.dtype)


def _compute_next_block_indices(kv_head_idx, logical_q_blk_idx, kv_blk_idx,
                                num_logical_q_blks, kv_blk_size, seq_ids,
                                effective_kv_lens_ref):
  """ Compute the next block indices.
  
      Given the current kv_head_idx, logical_q_blk_idx, kv_blk_idx, return the next_kv_head_idx, next_logical_q_blk_idx, next_kv_blk_idx.

      Note, k_pages has shape [num_kv_heads, total_num_pages, page_size, head_dim].
      To get the KV, it needs the kv_head_idx, then we need the sequence_idx
      and the kv_blk_idx to get the offset.
  """

  def advance_kv_head_idx():
    next_kv_head_idx = kv_head_idx + 1
    return next_kv_head_idx, 0, 0

  def advance_logical_q_blk_idx():
    next_logical_q_blk_idx = logical_q_blk_idx + 1
    return lax.cond(
        next_logical_q_blk_idx < num_logical_q_blks,
        lambda: (kv_head_idx, next_logical_q_blk_idx, 0),
        advance_kv_head_idx,
    )

  kv_blk_idx += 1
  cur_seq_idx = seq_ids[logical_q_blk_idx]
  effective_kv_len_cur_seq = effective_kv_lens_ref[cur_seq_idx]
  return lax.cond(
      kv_blk_idx * kv_blk_size < effective_kv_len_cur_seq,
      lambda: (kv_head_idx, logical_q_blk_idx, kv_blk_idx),
      advance_logical_q_blk_idx,
  )


def paged_flash_attention_kernel(
    # prefetch refs
    sequence_metadata_ref,  # Tuple (seq_ids, physical_q_tile_ids)
    num_logical_q_tiles_1d,
    effective_kv_lens_ref,  # [num_tokens]
    # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[num_tokens, pages_per_sequence]
    effective_cu_q_lens_ref,  # [num_tokens + 1]
    buffer_index_ref,
    step_ref,
    # kernel inputs
    # At caller, q.shape= [num_q_heads, num_tokens, head_dim]
    q_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    k_pages_hbm_ref,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    v_scales_pages_hbm_ref,
    cur_page_indices_ref,  # [num_kv_pages_per_block]
    next_page_indices_ref,  # [num_kv_pages_per_block]
    # same shape as q_ref: [1, num_q_heads_per_kv_head, num_queries_per_block, head_dim], output
    # outputs
    o_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    l_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    m_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    # scratch space
    k_vmem_buffer,  # (2, num_kv_pages_per_block, num_kv_heads, head_dim)
    k_scales_vmem_buffer,
    v_vmem_buffer,  # (2, num_kv_pages_per_block, num_kv_heads, head_dim)
    v_scales_vmem_buffer,
    sem,
    l_scratch_ref,
    m_scratch_ref,
    acc_scratch_ref,
    *,
    # The following parameters are not passed to Mosaic and not in SMEM. They are static values.
    pages_per_sequence: int,  # Note [bs, pages_per_sequence] = page_indices.shape
    num_tokens: int,
    num_seqs: int,
    num_kv_pages_per_block: int,
    mask_value: float,
    sm_scale: float,
):
  kv_head_idx, logical_q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
  )
  num_logical_q_blks = pl.num_programs(1)
  num_q_heads_per_kv_head, num_queries_per_block, head_dim = q_ref.shape
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages_hbm_ref.shape
  kv_blk_size = page_size * num_kv_pages_per_block

  seq_ids, physical_q_tile_ids = sequence_metadata_ref
  cur_seq_idx = seq_ids[logical_q_blk_idx]
  effective_kv_len_cur_seq = effective_kv_lens_ref[cur_seq_idx]
  should_run = (kv_blk_idx * kv_blk_size < effective_kv_len_cur_seq)

  @pl.when(should_run)
  def get_kv_and_run_flash_attention():

    def create_kv_async_copy_descriptors(seq_idx, kv_head_idx, kv_blk_idx,
                                         buffer_index, page_indices):
      pages_to_load = num_kv_pages_per_block
      async_copy_k = MultiPageAsyncCopyDescriptor(
          k_pages_hbm_ref,
          k_scales_pages_hbm_ref,
          k_vmem_buffer.at[buffer_index],
          k_scales_vmem_buffer.at[buffer_index]
          if k_scales_vmem_buffer is not None else None,
          sem,
          page_indices,  # [num_kv_pages_per_block]
          pages_to_load,
          kv_head_idx,
      )
      async_copy_v = MultiPageAsyncCopyDescriptor(
          v_pages_hbm_ref,
          v_scales_pages_hbm_ref,
          v_vmem_buffer.at[buffer_index],
          v_scales_vmem_buffer.at[buffer_index]
          if v_scales_vmem_buffer is not None else None,
          sem,
          page_indices,  # [num_kv_pages_per_block]
          pages_to_load,
          kv_head_idx,
      )
      return async_copy_k, async_copy_v

    step = step_ref[0]
    buffer_index = buffer_index_ref[0]

    @pl.when(step == 0)
    def prefetch_first_block():  # pylint: disable=unused-variable
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          cur_seq_idx, kv_head_idx, kv_blk_idx, buffer_index,
          cur_page_indices_ref)
      async_copy_k.start()
      async_copy_v.start()

    next_kv_head_idx, next_logical_q_blk_idx, next_kv_blk_idx = _compute_next_block_indices(
        kv_head_idx, logical_q_blk_idx, kv_blk_idx, num_logical_q_blks,
        kv_blk_size, seq_ids, effective_kv_lens_ref)

    @pl.when(next_kv_head_idx < num_kv_heads)
    def prefetch_next_block():  # pylint: disable=unused-variable
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      next_seq_idx = seq_ids[next_logical_q_blk_idx]
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_seq_idx, next_kv_head_idx, next_kv_blk_idx, next_buffer_index,
          next_page_indices_ref)
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index

    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        cur_seq_idx, kv_head_idx, kv_blk_idx, buffer_index,
        cur_page_indices_ref)
    k = async_copy_k.wait_and_get_loaded(
    )  # [pages_per_compute_block*page_size,head_dim]
    v = async_copy_v.wait_and_get_loaded()
    assert k.shape == (num_kv_pages_per_block * page_size, head_dim)
    assert v.shape == (num_kv_pages_per_block * page_size, head_dim)

    for q_head_idx in range(num_q_heads_per_kv_head):
      _flash_attention(
          q_head_idx,
          sequence_metadata_ref,
          effective_kv_lens_ref,
          effective_cu_q_lens_ref,
          # kernel inputs
          q_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
          k,
          v,
          # outputs
          o_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
          l_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
          m_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
          # scratch space
          l_scratch_ref,
          m_scratch_ref,
          acc_scratch_ref,
          num_tokens=num_tokens,
          num_seqs=num_seqs,
          num_kv_pages_per_block=num_kv_pages_per_block,
          num_queries_per_block=num_queries_per_block,
          mask_value=mask_value,
          page_size=page_size,
          head_dim=head_dim,
          num_q_heads_per_kv_head=num_q_heads_per_kv_head,
          sm_scale=sm_scale,
      )
    step_ref[0] = step + 1
    # end of get_kv_and_run_flash_attention


def _round_up_to_multiple_of_tm(x, tm):
  return (x + tm - 1) // tm * tm


MIN_BLOCK_SIZE = 128


@checkify.checkify
@functools.partial(
    jax.jit,
    static_argnames=[
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "mask_value",
        "num_seqs",
        "sm_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    v_pages: jax.Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    kv_lens: jax.Array,  # i32[num_tokens]
    page_indices: jax.Array,  # i32[num_tokens, pages_per_sequence]
    cu_q_lens: jax.Array,  # i32[num_tokens + 1]
    num_seqs,  # int
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    num_kv_pages_per_block: int = 128,
    num_queries_per_block: int = 128,
    sm_scale: float = 1.0,
) -> jax.Array:
  """Paged attention kernel with ragged input.

  Args:
    q: A [num_tokens, num_q_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    kv_lens: A i32[num_tokens] jax.Array the effective kv length of each 
      sequence. For example, if we have three sequences, lengths could be 
      [16, 3, 1024, x, x, x, x, ...] where x is any value for padding. While 
      lengths's shape is [num_tokens], only the first num_seqs values are valid.
      The rest should be ignored.
    page_indices: A i32[num_tokens, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`. Similar to kv_lens, only the first
      num_seqs values are valid.
    cu_q_lens: A i32[num_tokens+1] jax.Array the cumulative sum of the effective
      query lengths. Similar to kv_lens, only the first num_seqs+1 values are
      valid.
    num_seqs: the number of sequences.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    num_kv_pages_per_block: how many kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: how many queries to be processes in one flash
      attention block in the pallas kernel.

    The num_tokens, num_seqs, and pages_per_sequence are dynamic. If they are
      very dynamic, then the overhead could be high due to the recompilation.

  Returns:
    The output of attention([num_tokens, num_q_heads, head_dim]).
  """
  # TODO: consider remove the k_scales_pages and v_scales_pages during cleaning up.
  if isinstance(k_pages, quantization_utils.QuantizedTensor):
    k_pages, k_scales_pages = k_pages.weight, k_pages.scales
    assert isinstance(k_scales_pages, jax.Array)  # For typing.
    k_scales_pages = jnp.broadcast_to(
        k_scales_pages, (*k_scales_pages.shape[:-1], k_pages.shape[-1]))
  else:
    k_scales_pages = None
  if isinstance(v_pages, quantization_utils.QuantizedTensor):
    v_pages, v_scales_pages = v_pages.weight, v_pages.scales
    assert isinstance(v_scales_pages, jax.Array)  # For typing.
    v_scales_pages = jnp.broadcast_to(
        v_scales_pages, (*v_scales_pages.shape[:-1], v_pages.shape[-1]))
  else:
    v_scales_pages = None

  num_tokens, num_q_heads, head_dim = q.shape
  # Why the permute_dims is needed? Before permute, q.shape=[num_tokens, num_q_heads, head_dim]; then when we apply the GridSpec, the 2nd last dimension is num_q_heads which is hard to be a multiple of 8.
  # If permute_dims turns out to be expensive, try jnp.swapaxes. The compiler
  # may optimize the copies away.
  # Or consider unsqueeze a dimension at the 2nd last dimension and squeeze it
  # out later so that num_q_heads doesn't have to be the 2nd last dimension and hence doesn't subject to the multiple of 8 constraint.
  q = jnp.permute_dims(q, (1, 0, 2))  # [num_q_heads, num_tokens, head_dim]
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages.shape
  check_kernel_input(q, k_pages, v_pages, kv_lens, page_indices, cu_q_lens,
                     num_seqs, num_kv_pages_per_block)
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads

  # num_logical_q_tiles is a zero-dimensional array.
  sequence_metadata, num_logical_q_tiles = make_sequence_metadata(
      cu_q_lens=cu_q_lens[:num_seqs + 1],
      m=num_tokens,
      tm=num_queries_per_block,
      start_sequence=jnp.array([0]),
      num_sequences=num_seqs,
  )

  pages_per_sequence = page_indices.shape[1]
  num_kv_blks = pages_per_sequence // num_kv_pages_per_block
  grid = (num_kv_heads, num_logical_q_tiles, num_kv_blks)

  # out_shape
  o_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  l = jax.ShapeDtypeStruct((num_q_heads, num_tokens, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  m = jax.ShapeDtypeStruct((num_q_heads, num_tokens, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  out_shape = (o_shape, l, m)

  # in-spec. Note currently q.shape=[num_q_heads, num_tokens, head_dim]
  # Within the kernel, q.shape should be [num_q_heads_per_kv_head, q_block_size, head_dim]
  def qo_index_map(kv_head_idx, logical_q_blk_idx, kv_blk_idx,
                   sequence_metadata, *_):
    seq_ids, physical_q_tile_ids = sequence_metadata
    del seq_ids
    physical_q_blk_idx = physical_q_tile_ids[logical_q_blk_idx]
    return (kv_head_idx, physical_q_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_heads_per_kv_head, num_queries_per_block, head_dim),
      qo_index_map,
  )
  # Note page_indices.shape=[num_tokens, pages_per_sequence], pages_per_sequence % num_kv_pages_per_block==0
  # Unsqueeze an extra dimension in page_indices so that num_tokens can avoid the 2nd last dimension having to be a multiple of 8.
  expanded_page_indices = jnp.expand_dims(
      page_indices, 1)  # [num_tokens, 1, pages_per_sequence]

  def cur_page_indices_index_map(kv_head_idx, logical_q_blk_idx, kv_blk_idx,
                                 sequence_metadata, *_):
    seq_ids, physical_q_tile_ids = sequence_metadata
    del physical_q_tile_ids
    seq_id = seq_ids[logical_q_blk_idx]
    return (seq_id, 0, kv_blk_idx)

  cur_page_indices_spec = pl.BlockSpec(
      (None, None, num_kv_pages_per_block),
      cur_page_indices_index_map,
      memory_space=pltpu.TPUMemorySpace.SMEM,
  )

  page_size = k_pages.shape[2]
  kv_blk_size = page_size * num_kv_pages_per_block

  def next_kv_blk_page_indices_index_map(kv_head_idx, logical_q_blk_idx,
                                         kv_blk_idx, sequence_metadata,
                                         num_logical_q_tiles_1d, kv_lens, *_):
    seq_ids, physical_q_tile_ids = sequence_metadata
    next_kv_head_idx, next_logical_q_blk_idx, next_kv_blk_idx = _compute_next_block_indices(
        kv_head_idx, logical_q_blk_idx, kv_blk_idx, num_logical_q_tiles_1d[0],
        kv_blk_size, seq_ids, kv_lens)
    del physical_q_tile_ids
    next_seq_id = seq_ids[next_logical_q_blk_idx]
    return (next_seq_id, 0, next_kv_blk_idx)

  next_page_indices_spec = pl.BlockSpec(
      (None, None, num_kv_pages_per_block),
      next_kv_blk_page_indices_index_map,
      memory_space=pltpu.TPUMemorySpace.SMEM,
  )
  in_specs = [
      q_block_spec,
      # Below 4 correspond to the 4 input: k_pages, k_scales_pages, q_pages, q_scales_pages.
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      None,
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      None,
      cur_page_indices_spec,
      next_page_indices_spec,
  ]

  # out_spec
  # o_specs should be the same as q_block_spec
  o_specs = q_block_spec
  # lm_index_map is same as qo_index_map
  lm_index_map = qo_index_map
  out_specs = [
      o_specs,
      pl.BlockSpec(
          (num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE),
          lm_index_map),  # l
      pl.BlockSpec(
          (num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE),
          lm_index_map),  # m
  ]

  # scratch space. Note k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
  l_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE),
      jnp.float32)
  m_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE),
      jnp.float32)
  acc_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_block, head_dim), jnp.float32)
  scratch_shapes = [
      pltpu.VMEM(
          (
              2,  # For double buffering during DMA copies.
              num_kv_pages_per_block,
              page_size,
              head_dim,
          ),
          k_pages.dtype,
      ),  # k_pages buffer, k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
      None,  # k_scales_pages=None
      pltpu.VMEM(
          (
              2,  # For double buffering during DMA copies.
              num_kv_pages_per_block,
              page_size,
              head_dim,
          ),
          v_pages.dtype,
      ),  # v_pages buffer
      None,  # v_scales_pages=None
      pltpu.SemaphoreType.DMA,
      l_scratch,
      m_scratch,
      acc_scratch,
  ]

  kernel = pl.pallas_call(
      functools.partial(
          paged_flash_attention_kernel,
          pages_per_sequence=pages_per_sequence,
          num_tokens=num_tokens,
          num_seqs=num_seqs,
          num_kv_pages_per_block=num_kv_pages_per_block,
          mask_value=mask_value,
          sm_scale=sm_scale,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=6,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.TPUCompilerParams(
          # due to compute_block_indices, we loop kv_head, q_blk, kv_blk, the order matters.
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
      out_shape=out_shape,
  )
  buffer_index = jnp.zeros((1,), jnp.int32)
  step = jnp.zeros((1,), jnp.int32)
  # Why converting num_logical_q_tiles to a 1d array? It's due to "INTERNAL: Mosaic failed to compile TPU kernel: 0-rank memref not supported"
  num_logical_q_tiles_1d = jnp.array([num_logical_q_tiles])
  outputs = kernel(
      # prefetch
      sequence_metadata,
      num_logical_q_tiles_1d,
      kv_lens,
      cu_q_lens,
      buffer_index,
      step,
      # kernel inputs
      q,
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
      expanded_page_indices,  # for the current iteration
      expanded_page_indices,  # for the next iteration
  )
  ret = outputs[0]
  return jnp.permute_dims(ret, (1, 0, 2)).astype(q.dtype)
