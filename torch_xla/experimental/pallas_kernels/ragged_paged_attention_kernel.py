from collections.abc import Sequence
from collections import namedtuple
import functools
from typing import Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

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
  return SequenceMetadata(num_tiles, group_ids, m_tile_ids)

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
  if num_seqs > num_tokens:
    raise ValueError(f"num_seqs must be less or equal to num_tokens. Got {num_seqs} and {num_tokens}")
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

def paged_flash_attention_kernel(
    # prefetch refs
    effective_kv_lens_ref,  # [num_tokens]
    # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[num_tokens, pages_per_sequence]
    page_indices_1d_ref,
    effective_cu_q_lens_ref,  # [num_tokens + 1]
    buffer_index_ref,
    step_ref,
    # kernel inputs
    # At caller, q.shape=
    q_ref,  # q_ref.shape=[1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    k_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    v_scales_pages_hbm_ref,
    # same shape as q_ref: [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim], output
    # outputs
    o_ref,
    l_ref,
    m_ref,
    # scratch space
    k_vmem_buffer,  # (2, num_kv_pages_per_compute_block, num_kv_heads, head_dim)
    k_scales_vmem_buffer,
    v_vmem_buffer,  # (2, num_kv_pages_per_compute_block, num_kv_heads, head_dim)
    v_scales_vmem_buffer,
    sem,
    l_scratch_ref,
    m_scratch_ref,
    acc_scratch_ref,
    *,
    # Where do the following parameter live? SMEM?
    pages_per_sequence: int,  # Note [bs, pages_per_sequence] = page_indices.shape
    num_tokens: int,
    num_seqs: int,
    num_kv_pages_per_compute_block: int,
    mask_value: float,
):
  num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim = q_ref.shape
  o_ref[...] = jnp.zeros_like(q_ref)
  l_ref[...] = jnp.zeros_like(l_ref)
  m_ref[...] = jnp.zeros_like(m_ref)

MIN_BLOCK_SIZE = 128

# TODO(xw32): uncomment this once the kernel output is correct.
# @functools.partial(
#     jax.jit,
#     static_argnames=[
#         "num_kv_pages_per_compute_block",
#         "num_queries_per_compute_block",
#         "mask_value",
#     ],
# )
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
    num_kv_pages_per_compute_block: int = 16,
    num_queries_per_compute_block: int = 128,
) -> jax.Array:
  """Paged grouped query attention.

  Args:
    q: A [num_tokens, num_q_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    kv_lens: A i32[num_tokens] jax.Array the effective kv length of each sequence.
    page_indices: A i32[num_tokens, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    cu_q_lens: A i32[num_tokens+1] jax.Array the cumulative sum of the effective
      query lengths.
    num_seqs: A i32[] jax.Array the number of sequences.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    num_kv_pages_per_compute_block: how many kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_compute_block: how many queries to be processes in one flash
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
  q = jnp.permute_dims(q, (1, 0, 2))  # [num_q_heads, num_tokens, head_dim]
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages.shape
  check_kernel_input(q, k_pages, v_pages,kv_lens, page_indices, cu_q_lens, num_seqs, num_kv_pages_per_compute_block)
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads

  num_logical_q_tiles, seq_ids, physical_q_tile_ids = make_group_metadata(
      cu_q_lens=cu_q_lens,
      m=num_tokens,
      tm=num_queries_per_compute_block,
      start_group=jnp.array([0]),
      num_seqs=num_seqs,
  )

  pages_per_sequence = page_indices.shape[1]
  num_kv_blks = pages_per_sequence // num_kv_pages_per_compute_block
  # note, num_logical_q_tiles has type jnp.ndarray
  grid = (num_kv_heads, num_logical_q_tiles.item(), num_kv_blks)
  print(f"xw32 line367 grid={grid}")

  # out_shape
  o_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  # xw32: need to double check that the out_shape of l and m are correct.
  l = jax.ShapeDtypeStruct((num_q_heads, num_tokens, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  m = jax.ShapeDtypeStruct((num_q_heads, num_tokens, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  out_shape = (o_shape, l, m)
  print(f'xw32 {out_shape=}')

  # in-spec. Note currently q.shape=[num_q_heads, num_tokens, head_dim]
  # Within the kernel, q.shape should be [num_q_heads_per_kv_head, q_block_size, head_dim]
  def qo_index_map(kv_head_idx, logical_q_idx, kv_blk_idx, *_):
    physical_q_tile_id = physical_q_tile_ids[logical_q_idx]
    return (kv_head_idx, physical_q_tile_id, 0)
  q_block_spec = pl.BlockSpec(
    (num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim),
    qo_index_map,
  )
  q_dtype_for_kernel_launch = q.dtype
  print(f'xw32 {q_block_spec=}')
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
  o_specs = pl.BlockSpec(  # Should be the same as q_block_spec
    (num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim),
    qo_index_map,
  )

  # lm_index_map is same as qo_index_map
  def lm_index_map(kv_head_idx, logical_q_idx, kv_blk_idx, *_):
    return (kv_head_idx, logical_q_idx, 0)

  out_specs = [
      o_specs,
      pl.BlockSpec(
          (num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
          lm_index_map),  # l
      pl.BlockSpec(
          (num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
          lm_index_map),  # m
  ]

  # scratch space. Note k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
  l_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
      jnp.float32)
  m_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
      jnp.float32)
  acc_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim),
      jnp.float32)
  scratch_shapes = [
      pltpu.VMEM(
          (
              2,  # For double buffering during DMA copies.
              num_kv_pages_per_compute_block,
              page_size,
              head_dim,
          ),
          k_pages.dtype,
      ),  # k_pages buffer, k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
      None,  # k_scales_pages=None
      pltpu.VMEM(
          (
              2,  # For double buffering during DMA copies.
              num_kv_pages_per_compute_block,
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
          num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
          mask_value=mask_value,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=5,  # TODO(xw32): may need to adjust.
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
  outputs = kernel(
      # prefetch
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
  print(f"xw32 line495 ret.shape={ret.shape}, {ret=}")
  return jnp.permute_dims(ret, (1, 0, 2)).astype(q.dtype)
