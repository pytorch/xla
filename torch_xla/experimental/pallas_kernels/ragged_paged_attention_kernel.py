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
      page_indices,
      page_indices_start_offset,
      num_pages_to_load,
      kv_head_index,
  ):
    # Original k_pages has shape [num_kv_heads, total_num_pages, page_size, head_dim]
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
    self._page_indices_start_offset = page_indices_start_offset
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
    page_index = self._page_indices[self._page_indices_start_offset + i]
    return pltpu.make_async_copy(self._pages_hbm_ref.at[page_index],
                                 self._vmem_buffer.at[i], self._sem)

  def _make_scales_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
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

SequenceMetadata = namedtuple(
    "SequenceMetadata",
    [
        "num_logical_q_tiles",
        "seq_ids",
        "physical_q_tile_ids",
    ],
)

GroupMetadata = Any

# https://github.com/jax-ml/jax/blob/9fb29766a2130e74a85cba30420cf777d185ea5a/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L79
# TODO(xw32): need to do some renaming to adapt to our case.
# Currently, group maps to sequence.
def make_group_metadata(
    *,
    cu_q_lens: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_seqs: int,
):
  """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    start_group: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
      combination with group_offset.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, jnp.ndarray with shape [num_groups+1] and jnp.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute.
  """
  num_groups = num_seqs
  end_group = start_group + num_seqs - 1

  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = cu_q_lens[1:]
  group_offsets = cu_q_lens

  # Assign a group id to each grid index.
  #
  # If a group starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each group by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.

  # (1) Round the group_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change group_offsets[num_groups], which is m
  # (because we enforce m is divisible by tm).
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
  print('xw32 {rounded_group_ends=}')

  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts

  # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
  # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' group never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no group has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every group has a partial except the 0-th group, the total
  # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
  #
  # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  group_tiles = rounded_group_sizes // tm

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)
  group_ids = jnp.repeat(
      jnp.arange(num_groups, dtype=jnp.int32),
      group_tiles[:num_groups], # would it introduce dynamic shape to impact JIT?
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the group that owns the tile.
  # The remaining possible visits occur when a group starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each group starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the group that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram. Also filter out any
  # group which is empty.
  #
  # TODO(tgale): Invert the 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = ((group_offsets[:-1] % tm) == 0)

  partial_tile_ids = jnp.where(
      partial_tile_mask, tiles_m, group_offsets[:-1] // tm
  )

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0]
      + 1
  )

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = jnp.repeat(
      jnp.arange(tiles_m, dtype=jnp.int32),
      tile_visits.astype(jnp.int32),
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Account for sharding.
  #
  # Find the start of the groups owned by our shard and shift the group_ids and
  # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
  #
  # TODO(tgale): Move this offset into the kernel to avoid these rolls.
  first_tile_in_shard = (group_ids < start_group).sum()
  group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

  # Calculate the number of tiles we need to compute for our shard.
  #
  # Remove tile visits that belong to a group not in our shard.
  iota = jnp.arange(num_groups, dtype=jnp.int32)
  active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles[:num_groups], 0)
  num_tiles = group_tiles.sum()
  return (group_ids, m_tile_ids), num_tiles  # num_logical_q_tiles, seq_ids, physical_q_tile_ids

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
    raise ValueError("kv_lens.shape[0] must be thet same as num_tokens. Got"
                     f" {kv_lens.shape[0]} and {num_tokens}")
  if page_indices.shape[0] != num_tokens:
    raise ValueError("page_indices.shape[0] must be thet same as num_tokens. Got"
                     f" {page_indices.shape[0]} and {num_tokens}")
  if cu_q_lens.shape[0] != num_tokens + 1:
    raise ValueError("cu_q_lens.shape[0] must be thet same as num_tokens + 1. Got"
                     f" {cu_q_lens.shape[0]} and {num_tokens + 1}")
  for i in range(num_seqs):
    cur_q_len = cu_q_lens[i+1] - cu_q_lens[i]
    cur_kv_len = kv_lens[i]
    jax.debug.print("xw32 line308 {i} {cur_q_len}, {cur_kv_len}", i=i, cur_q_len=cur_q_len, cur_kv_len=cur_kv_len)
    checkify.check(cur_q_len <= cur_kv_len, "cur_q_len must be less or equal to cur_kv_len. Got {} and {}", cur_q_len, cur_kv_len)
  if num_seqs > num_tokens:
    raise ValueError(f"num_seqs must be less or equal to num_tokens. Got {num_seqs} and {num_tokens}")
  # int16: will pack. need to explicit cast to int32. int64 is not supported in Pallas. for smem 1d case.
  # 2d smem: int16 will be packed with an empty. So we didn't save any memory.
  # scalar: use i32 (1, N). int16 for (1, N) will be padding. Need to use (2, N).
  if kv_lens.dtype != jnp.int32 or page_indices.dtype != jnp.int32 or cu_q_lens.dtype != jnp.int32:
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {kv_lens.dtype=}, "
        f"{page_indices.dtype=}, {cu_q_lens.dtype=}")
  if num_kv_pages_per_block > pages_per_sequence:
    raise ValueError(
        f"{num_kv_pages_per_block=} should be smaller or equal to {pages_per_sequence=}"
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
    grid_id: jnp.ndarray,
    group_offsets: jnp.ndarray,
    group_ids: jnp.ndarray,
    m_tile_ids: jnp.ndarray,
    tm: int,
    tn: int,
) -> jnp.ndarray:
  """Mask for rows that belong to the current group in the current tile."""
  group_id = group_ids[grid_id]
  group_start = group_offsets[group_id]
  group_end = group_offsets[group_id + 1]
  m_id = m_tile_ids[grid_id] * tm
  iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
  return jnp.logical_and(iota >= group_start, iota < group_end)

def _flash_attention(
    q_head_idx_per_kv,  # scalar, ranges from 0 to num_query_heads_per_kv_head
    group_metadata_ref,  # (seq_ids, physical_q_tile_ids)
    effective_kv_lens_ref,  # [num_tokens]
    effective_cu_q_lens_ref,  # [num_tokens + 1]
    # kernel inputs
    q_ref,  # q_ref.shape=[num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    k,  # [kv_blk_size, head_dim]
    v,  # [kv_blk_size, head_dim]
    # outputs
    o_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    l_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    m_ref,  # [num_q_heads_per_kv_head, num_queries_per_block, MIN_BLOCK_SIZE]
    # scratch space
    # TODO: double check if the scratch ref shape is correct.
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
):
  assert q_ref.shape == (num_q_heads_per_kv_head, num_queries_per_block, head_dim)
  kv_blk_size = page_size * num_kv_pages_per_block
  assert k.shape == (kv_blk_size, head_dim)
  assert v.shape == (kv_blk_size, head_dim)
  
  kv_head_idx, logical_q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
  )
  seq_ids, physical_q_tile_ids = group_metadata_ref
  
  # If the q-dim physical tile is changed (meaning it is a new physical q-dim tile that has not visited before), initialize the acc_scratch_ref, m_scratch_ref, and l_scratch_ref to run the flash attention v2 algorithm.
  prev_logical_q_blk_idx = jnp.where(logical_q_blk_idx > 0, logical_q_blk_idx - 1, 0)
  is_first_processed_logical_q_blk = logical_q_blk_idx == 0
  physical_q_blk_changed = (physical_q_tile_ids[logical_q_blk_idx] != physical_q_tile_ids[prev_logical_q_blk_idx])
  first_time_seeing_physical_q_blk = jnp.logical_or(is_first_processed_logical_q_blk, physical_q_blk_changed)
  is_first_kv_blk = (kv_blk_idx == 0)
  should_init_scratch_ref = jnp.logical_and(is_first_kv_blk,
      first_time_seeing_physical_q_blk)
  @pl.when(should_init_scratch_ref)
  def init_scratch_ref():  # pylint: disable=unused-variable
    pl.debug_print("xw32 should_init_scratch_ref begins: kv_head_idx={}, logical_q_blk_idx={}, kv_blk_idx={}", kv_head_idx, logical_q_blk_idx, kv_blk_idx)
    l_scratch_ref[q_head_idx_per_kv] = jnp.zeros(
        l_scratch_ref[q_head_idx_per_kv].shape, jnp.float32)
    m_scratch_ref[q_head_idx_per_kv] = jnp.full(
        m_scratch_ref[q_head_idx_per_kv].shape, -jnp.inf, jnp.float32)
    acc_scratch_ref[q_head_idx_per_kv] = jnp.zeros(
        acc_scratch_ref[q_head_idx_per_kv].shape, jnp.float32)
  
  m_prev = m_scratch_ref[q_head_idx_per_kv]  # [num_queries_per_block, MIN_BLOCK_SIZE]
  l_prev = l_scratch_ref[q_head_idx_per_kv]  # [num_queries_per_block, MIN_BLOCK_SIZE]
  
  # Load the whole q_block that belongs to the current physical q_blk and compute the attention. When we write, we only write the part that belongs to the current sequence.
  # I cannot just load only the part of q_block that belongs to the current sequence, because it results in dynamic shapes and then fails the JIT compilation.
  # Note, q_ref.shape=[num_q_heads_per_kv_head, num_queries_per_block, head_dim]
  q = q_ref[q_head_idx_per_kv, :, :].astype(jnp.float32)  # [block_q, head_dim]
  assert q.shape == (num_queries_per_block, head_dim)
  s = jnp.einsum(
      'qd,td->qt', q, k,
      preferred_element_type=jnp.float32)  # [block_q, block_k]
  assert s.shape == (num_queries_per_block, kv_blk_size)
  
  # Modify the mask accordingly: first form the mask. Then move the mask down to the right place.
  cur_seq_idx = seq_ids[logical_q_blk_idx]
  cur_seq_start = effective_cu_q_lens_ref[cur_seq_idx]
  cur_seq_end = effective_cu_q_lens_ref[cur_seq_idx+1]
  physical_q_blk_idx = physical_q_tile_ids[logical_q_blk_idx]
  seq_start_in_cur_physical_q_blk = cur_seq_start >= physical_q_blk_idx*num_queries_per_block
  # seq_start_idx_in_cur_physical_q_blk = jnp.where(seq_start_in_cur_physical_q_blk,
  #                                             cur_seq_start - physical_q_blk_idx*num_queries_per_block,
  #                                             0)
  # q_index = physical_q_blk_idx*num_queries_per_block - seq_start_idx_in_cur_physical_q_blk  # start_q_idx_for_cur_seq_in_cur_physical_q_blk. TODO: let's rename num_queries_per_block to q_blk_size later.
  q_index = physical_q_blk_idx*num_queries_per_block-cur_seq_start
  pl.debug_print("xw32 line423, kv_head_idx={}, logical_q_blk_idx={}, kv_blk_idx={}, q_index={}", kv_head_idx, logical_q_blk_idx, kv_blk_idx, q_index)
  kv_index = kv_blk_idx * kv_blk_size
  effective_kv_len = effective_kv_lens_ref[cur_seq_idx]
  effective_q_len = cur_seq_end - cur_seq_start
  row_ids = (
      effective_kv_len - effective_q_len) + q_index + jax.lax.broadcasted_iota(
          jnp.int32,
          (num_queries_per_block, kv_blk_size), 0)
  col_ids = kv_index + jax.lax.broadcasted_iota(
      jnp.int32,
      (num_queries_per_block, kv_blk_size), 1)
  causal_mask = jnp.where(row_ids < col_ids, mask_value, 0.) # TODO: use this mask.
  # causal_mask_debug = jnp.where(row_ids < col_ids, -1, 0) # TODO: remove this line.
  should_print_mask = jnp.logical_and(kv_head_idx==0, logical_q_blk_idx==2)
  # @pl.when(should_print_mask)
  # def print_mask():  # pylint: disable=unused-variable
  #   pl.debug_print("xw32 line438, causal_mask={}", causal_mask)
  assert causal_mask.shape == (num_queries_per_block,
                               kv_blk_size)
  s = s + causal_mask  # [block_q, block_k]
  assert s.shape == (num_queries_per_block,
                     kv_blk_size)
  
  m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
  # why the second dim of m_prev, m_curr, or m_next is 128?
  m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].
  
  block_k_repeats, rem = divmod(kv_blk_size, MIN_BLOCK_SIZE)
  if rem:
    raise NotImplementedError(
        f"{kv_blk_size=} should be a multiple of {MIN_BLOCK_SIZE}"
    )
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
  lm_mask = _get_store_mask(grid_id=logical_q_blk_idx,
                            group_offsets=effective_cu_q_lens_ref,
                            group_ids=seq_ids,
                            m_tile_ids=physical_q_tile_ids,
                            tm=num_queries_per_block,
                            tn=MIN_BLOCK_SIZE,
                            )
  # Should I use jax.lax.select or jnp.where? What's the difference? eg: jnp.where(lm_mask, l_next, 0), jnp.where(lm_mask, m_next, 0)
  # Can `lm_mask[...]` be `lm_mask`?
  l_scratch_ref[q_head_idx_per_kv] = jax.lax.select(lm_mask[...], l_next, l_scratch_ref[q_head_idx_per_kv])
  m_scratch_ref[q_head_idx_per_kv] = jax.lax.select(lm_mask[...], m_next, m_scratch_ref[q_head_idx_per_kv])
  
  # @pl.when(should_print_mask)
  # def _():  # pylint: disable=unused-variable
  #   print("xw32 line492, l_next.shape={}, ", l_next.shape)
  #   pl.debug_print("xw32 line492, l_next[6]={}", l_next[6])
  l_next_inv_safe = jnp.where(l_next == 0.0, 1.0,
                              1.0 / l_next)  # [block_q, 128]
  temp = acc_scratch_ref[q_head_idx_per_kv] * l_broadcast(l_corr * l_next_inv_safe)
  acc_mask = _get_store_mask(grid_id=logical_q_blk_idx,
                            group_offsets=effective_cu_q_lens_ref,
                            group_ids=seq_ids,
                            m_tile_ids=physical_q_tile_ids,
                            tm=num_queries_per_block,
                            tn=head_dim,
                            )
  print(f"xw32 line486 {acc_mask.shape=}, {temp.shape=}, {acc_scratch_ref[q_head_idx_per_kv]=}")
  acc_scratch_ref[q_head_idx_per_kv] = jax.lax.select(acc_mask[...], temp, acc_scratch_ref[q_head_idx_per_kv])
  # Note Matmul operandlhs must have a shape divisible by (16, 1)
  o_curr = jax.lax.dot(
      p.astype(v.dtype), v,
      preferred_element_type=jnp.float32)  # [block_q, 128]
  temp = (acc_scratch_ref[q_head_idx_per_kv] + o_curr * l_broadcast(l_next_inv_safe))
  # @pl.when(should_print_mask)
  # def _():  # pylint: disable=unused-variable
  #   print("xw32 line512, temp.shape={}", temp.shape)
  #   pl.debug_print("xw32 line512, temp={}", temp)
  acc_scratch_ref[q_head_idx_per_kv] = jax.lax.select(acc_mask[...], temp, acc_scratch_ref[q_head_idx_per_kv])
  
  # Store the result from VMEM to HBM only when it is the last kv_block and the next q-dim logical tile belongs to a different q-dim physical tile.
  is_last_kv_blk_idx = (kv_blk_idx == (pl.cdiv(effective_kv_len, kv_blk_size) - 1))
  num_logical_q_blks = pl.num_programs(1)  # grid=(num_kv_heads, num_logical_q_tiles, num_kv_blks)
  next_logical_q_blk_idx = jnp.where(logical_q_blk_idx == num_logical_q_blks - 1,
                                     logical_q_blk_idx,
                                     logical_q_blk_idx+1)
  is_last_logical_q_blk = (logical_q_blk_idx == num_logical_q_blks-1)
  physical_q_blk_will_change = (physical_q_tile_ids[logical_q_blk_idx] != physical_q_tile_ids[next_logical_q_blk_idx])
  last_time_seeing_cur_physical_q_blk = jnp.logical_or(is_last_logical_q_blk, physical_q_blk_will_change)
  should_store_to_hbm = jnp.logical_and(is_last_kv_blk_idx, last_time_seeing_cur_physical_q_blk)
  @pl.when(should_store_to_hbm)
  def store_to_hbm():  # pylint: disable=unused-variable
    pl.debug_print("xw32 store_to_hbm begins: kv_head_idx={}, logical_q_blk_idx={}, kv_blk_idx={}", kv_head_idx, logical_q_blk_idx, kv_blk_idx)
    o_ref[q_head_idx_per_kv] = acc_scratch_ref[q_head_idx_per_kv].astype(
        o_ref.dtype)
    l_ref[q_head_idx_per_kv] = l_scratch_ref[q_head_idx_per_kv].astype(
        l_ref.dtype)
    m_ref[q_head_idx_per_kv] = m_scratch_ref[q_head_idx_per_kv].astype(
        m_ref.dtype)


def paged_flash_attention_kernel(
    # prefetch refs, in smem
    group_metadata_ref,  # (seq_ids, physical_q_tile_ids)
    effective_kv_lens_ref,  # [num_tokens]
    # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[num_tokens, pages_per_sequence]
    page_indices_1d_ref,
    effective_cu_q_lens_ref,  # [num_tokens + 1]
    buffer_index_ref,
    step_ref,
    # kernel inputs
    # At caller, q.shape= [num_q_heads, num_tokens, head_dim]
    q_ref,  # q_ref.shape=[num_q_heads_per_kv_head, num_queries_per_block, head_dim]
    k_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    v_scales_pages_hbm_ref,
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
    # Where do the following parameter live? SMEM? Not in smem. Not to pass in mosaic. Static value.
    pages_per_sequence: int,  # Note [bs, pages_per_sequence] = page_indices.shape
    num_tokens: int,
    num_seqs: int,
    num_kv_pages_per_block: int,
    mask_value: float,
):
  # assert the input shapes
  print(f"xw32 line283 paged_flash_attention_kernel begins. q_ref.shape={q_ref.shape}")
  kv_head_idx, logical_q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
  )
  num_logical_q_blks = pl.num_programs(1)
  num_q_heads_per_kv_head, num_queries_per_block, head_dim = q_ref.shape
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages_hbm_ref.shape
  kv_blk_size = page_size * num_kv_pages_per_block
  
  seq_ids, physical_q_tile_ids = group_metadata_ref
  cur_seq_idx = seq_ids[logical_q_blk_idx]
  effective_kv_len_cur_seq = effective_kv_lens_ref[cur_seq_idx]
  should_run = (kv_blk_idx * kv_blk_size < effective_kv_len_cur_seq)
  pl.debug_print("xw32 paged_flash_attention_kernel begins kv_head_idx={}, logical_q_blk_idx={}, kv_blk_idx={}, cur_seq_idx={}, effective_kv_len_cur_seq={}", kv_head_idx, logical_q_blk_idx, kv_blk_idx, cur_seq_idx, effective_kv_len_cur_seq)  # pl.debug_print can only print JAX type. So cannot print tuple such as q.shape.
  
  @pl.when(should_run)
  def get_kv_and_run_flash_attention():
    # grid = (num_kv_heads, num_logical_q_tiles, num_kv_blks)
    def compute_block_indices(kv_head_idx, logical_q_blk_idx, kv_blk_idx):
      """Return next_kv_head_idx, next_logical_q_blk_idx, next_kv_blk_idx
      
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
      
      cur_seq_idx = seq_ids[logical_q_blk_idx]
      effective_kv_len_cur_seq = effective_kv_lens_ref[cur_seq_idx]
      return lax.cond(
          kv_blk_idx*kv_blk_size < effective_kv_len_cur_seq,
          lambda: (kv_head_idx, logical_q_blk_idx, kv_blk_idx),
          advance_logical_q_blk_idx,
      )
      
    def create_kv_async_copy_descriptors(seq_idx, kv_head_idx, kv_blk_idx,
                                         buffer_index):
      page_offset = seq_idx * pages_per_sequence + kv_blk_idx * num_kv_pages_per_block
      pages_to_load = num_kv_pages_per_block
      async_copy_k = MultiPageAsyncCopyDescriptor(
          k_pages_hbm_ref,
          k_scales_pages_hbm_ref,
          k_vmem_buffer.at[buffer_index],
          k_scales_vmem_buffer.at[buffer_index]
          if k_scales_vmem_buffer is not None else None,
          sem,
          page_indices_1d_ref,  # [batch_size*pages_per_sequence]
          page_offset,
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
          page_indices_1d_ref,
          page_offset,
          pages_to_load,
          kv_head_idx,
      )
      return async_copy_k, async_copy_v

    step = step_ref[0]
    buffer_index = buffer_index_ref[0]

    @pl.when(step == 0)
    def prefetch_first_block():  # pylint: disable=unused-variable
      pl.debug_print("xw32 prefetch_first_block kv_head_idx={}, cur_seq_idx={}, kv_blk_idx={}, buffer_index={}", kv_head_idx, cur_seq_idx, kv_blk_idx, buffer_index)
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          cur_seq_idx, kv_head_idx, kv_blk_idx, buffer_index)
      async_copy_k.start()
      async_copy_v.start()

    # kv_head_idx, logical_q_blk_idx, kv_blk_idx
    next_kv_head_idx, next_logical_q_blk_idx, next_kv_blk_idx = compute_block_indices(kv_head_idx, logical_q_blk_idx, kv_blk_idx+1)
    
    @pl.when(next_kv_head_idx < num_kv_heads)
    def prefetch_next_block():  # pylint: disable=unused-variable
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      next_seq_idx = seq_ids[next_logical_q_blk_idx]
      pl.debug_print("xw32 prefetch_next_block next_kv_head_idx={}, next_seq_idx={}, next_kv_blk_idx={}, buffer_index={}", next_kv_head_idx, next_seq_idx, next_kv_blk_idx, next_buffer_index)
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_seq_idx, next_kv_head_idx, next_kv_blk_idx, next_buffer_index)
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index
          
    # xw32: is the async_copy_k and async_copy_v the same as the ones created in prefetch_first_block?
    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        cur_seq_idx, kv_head_idx, kv_blk_idx, buffer_index)
    k = async_copy_k.wait_and_get_loaded(
    )  # [pages_per_compute_block*page_size,head_dim]
    v = async_copy_v.wait_and_get_loaded()
    assert k.shape == (num_kv_pages_per_block*page_size, head_dim)
    assert v.shape == (num_kv_pages_per_block*page_size, head_dim)
    
    for q_head_idx in range(num_q_heads_per_kv_head):
      _flash_attention(
          q_head_idx,
          group_metadata_ref,
          effective_kv_lens_ref,
          effective_cu_q_lens_ref,
          # kernel inputs
          q_ref,  # q_ref.shape=[num_q_heads_per_kv_head, num_queries_per_block, head_dim]
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
      )
    step_ref[0] = step + 1
    # end of get_kv_and_run_flash_attention


MIN_BLOCK_SIZE = 128

# TODO(xw32): uncomment this once the kernel output is correct.
@checkify.checkify
@functools.partial(
    jax.jit,
    static_argnames=[
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "mask_value",
        "num_seqs",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    v_pages: jax.Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
    kv_lens: jax.Array,  # i32[num_tokens]
    page_indices: jax.Array,  # i32[num_tokens, pages_per_sequence]
    cu_q_lens: jax.Array,  # i32[num_tokens + 1]
    num_seqs,  # i32[]
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    num_kv_pages_per_block: int = 16,
    num_queries_per_block: int = 128,
) -> jax.Array:
  """Paged attention kernel with ragged input.

  Args:
    q: A [num_tokens, num_q_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    kv_lens: A i32[num_tokens] jax.Array the effective kv length of each 
      sequence. For example, if we have three sequences, lengths could be 
      [16, 3, 1024, x, x, x, x, ...] where x is any value for padding. While 
      lengths’s shape is [num_tokens], only the first num_seqs values are valid.
      The rest should be ignored.
    page_indices: A i32[num_tokens, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`. Similar to kv_lens, only the first
      num_seqs values are valid.
    cu_q_lens: A i32[num_tokens+1] jax.Array the cumulative sum of the effective
      query lengths. Similar to kv_lens, only the first num_seqs+1 values are
      valid.
    num_seqs: A i32[] jax.Array the number of sequences.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    num_kv_pages_per_block: how many kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: how many queries to be processes in one flash
      attention block in the pallas kernel.

  Returns:
    The output of attention([num_tokens, query_len, num_q_heads, head_dim]).
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
  # If permute_dims turns out to be expensive, try jnp.swapaxes. The compiler
  # may optimize the copies away.
  # Or consider unsqueeze a dimension at the 2nd last dimension and squeeze it 
  # out later.
  # jevin: can we not do the permute_dims?
  # Why the permute_dims is needed? Before permute, q.shape=[num_tokens, num_q_heads, head_dim]; then when we apply the GridSpec, the 2nd last dimension is num_q_heads which is hard to be a multiple of 8.
  q = jnp.permute_dims(q, (1, 0, 2))  # [num_q_heads, num_tokens, head_dim]
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages.shape
  check_kernel_input(q, k_pages, v_pages,kv_lens, page_indices, cu_q_lens, num_seqs, num_kv_pages_per_block)
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads

  group_metadata, num_logical_q_tiles = make_group_metadata(
      cu_q_lens=cu_q_lens,
      m=num_tokens,
      tm=num_queries_per_block,
      start_group=jnp.array([0]),
      num_seqs=num_seqs,
  )
  seq_ids, physical_q_tile_ids = group_metadata
  pl.debug_print("xw32 line797 seq_ids={}, physical_q_tile_ids={}, num_logical_q_tiles={}", seq_ids, physical_q_tile_ids, num_logical_q_tiles)

  pages_per_sequence = page_indices.shape[1]
  num_kv_blks = pages_per_sequence // num_kv_pages_per_block
  # num_logical_q_tiles has type jnp.ndarray. So we need the .item() below.
  grid = (num_kv_heads, num_logical_q_tiles, num_kv_blks)
  print(f"xw32 line367 grid={grid}")

  # out_shape
  o_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  # xw32: need to double check that the out_shape of l and m are correct.
  l = jax.ShapeDtypeStruct((num_q_heads, num_tokens, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  m = jax.ShapeDtypeStruct((num_q_heads, num_tokens, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  out_shape = (o_shape, l, m)

  # in-spec. Note currently q.shape=[num_q_heads, num_tokens, head_dim]
  # Within the kernel, q.shape should be [num_q_heads_per_kv_head, q_block_size, head_dim]
  def qo_index_map(kv_head_idx, logical_q_blk_idx, kv_blk_idx, group_metadata, *_):
    seq_ids, physical_q_tile_ids = group_metadata
    del seq_ids
    physical_q_blk_idx = physical_q_tile_ids[logical_q_blk_idx]
    return (kv_head_idx, physical_q_blk_idx, 0)
  q_block_spec = pl.BlockSpec(
    (num_q_heads_per_kv_head, num_queries_per_block, head_dim),
    qo_index_map,
  )
  q_dtype_for_kernel_launch = q.dtype
  in_specs = [
      q_block_spec,
      # Below 4 correspond to the 4 input: k_pages, k_scales_pages, q_pages, q_scales_pages.
      # TODO: consider to remove the k_scales_pages and v_scales_pages during cleaning up.
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      None,
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      None,
  ]

  # out_spec
  # jevin: create a qo spec and reuse it.
  o_specs = pl.BlockSpec(  # Should be the same as q_block_spec
    (num_q_heads_per_kv_head, num_queries_per_block, head_dim),
    qo_index_map,
  )

  # lm_index_map is same as qo_index_map
  # TODO: think about reusing q_indx_map.
  def lm_index_map(kv_head_idx, logical_q_blk_idx, kv_blk_idx, group_metadata, *_):
    seq_ids, physical_q_tile_ids = group_metadata
    del seq_ids
    physical_q_blk_idx = physical_q_tile_ids[logical_q_blk_idx]
    return (kv_head_idx, physical_q_blk_idx, 0)

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
      (num_q_heads_per_kv_head, num_queries_per_block, head_dim),
      jnp.float32)
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
          num_seqs=num_seqs,  # it they changes, need to recompile.
          num_kv_pages_per_block=num_kv_pages_per_block,
          mask_value=mask_value,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=6,  # TODO(xw32): may need to adjust.
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.TPUCompilerParams(
          # due to compute_block_indices, we loop batch, kv_head, q_blk, kv_blk, the order matters.
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
              "arbitrary",
          )),
      out_shape=out_shape,
  )
  # TODO: need to slice the page_indices later to avoid the SMEM OOM.
  page_indices_1d = page_indices.reshape(-1)
  buffer_index = jnp.zeros((1,), jnp.int32)
  step = jnp.zeros((1,), jnp.int32)

  # debug compile begins
  # To enable debug, uncomment this section, comment out the `kernel()` below and comment out the jax.jit above.
  # compiled_kernel = (
  #     jax.jit(kernel)
  #     .lower(
  #         # prefetch
  #         group_metadata,
  #         kv_lens,
  #         page_indices_1d,
  #         cu_q_lens,
  #         buffer_index,
  #         step,
  #         # kernel inputs
  #         q.astype(q_dtype_for_kernel_launch),  # TODO: do we need the `.astype`? Need to double check.
  #         k_pages,
  #         k_scales_pages,
  #         v_pages,
  #         v_scales_pages,
  #     )
  #     .compile({'xla_tpu_enable_log_recorder': 'true'})
  # )
  # outputs = compiled_kernel(
  #     # prefetch
  #     group_metadata,
  #     kv_lens,
  #     page_indices_1d,
  #     cu_q_lens,
  #     buffer_index,
  #     step,
  #     # kernel inputs
  #     q.astype(q_dtype_for_kernel_launch),  # TODO: do we need the `.astype`? Need to double check.
  #     k_pages,
  #     k_scales_pages,
  #     v_pages,
  #     v_scales_pages,
  # )
  # debug compile ends
  
  outputs = kernel(
      # prefetch
      group_metadata,
      kv_lens,
      page_indices_1d,
      cu_q_lens,
      buffer_index,
      step,
      # kernel inputs
      q.astype(q_dtype_for_kernel_launch),  # TODO: do we need the `.astype`? Need to double check.
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
  )
  ret = outputs[0]
  # print(f"xw32 line495 ret.shape={ret.shape}, {ret=}")
  return jnp.permute_dims(ret, (1, 0, 2)).astype(q.dtype)
