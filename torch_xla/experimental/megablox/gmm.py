"""Grouped matrix multiplication kernels for TPU written in Pallas."""

from typing import Any, Callable, Optional, Union
from torch_xla.experimental.megablox import common
from torch_xla.experimental.custom_kernel import jax_import_guard
import torch
import torch_xla
import numpy as np


def _validate_args(
    *,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    expected_rhs_dims: int = 3,
) -> 'tuple[jnp.ndarray, jnp.ndarray, jnp.dtype]':
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  """Validates the arguments for the gmm function."""
  # Validate 'lhs'.
  if lhs.dim() != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.dim()}-tensor.")
  common.assert_is_supported_dtype(lhs.dtype)

  # Validate 'rhs'.
  if rhs.dim() != expected_rhs_dims:
    raise ValueError(f"Expected {expected_rhs_dims}-tensor for 'rhs' but got"
                     f" {rhs.dim()}-tensor.")
  common.assert_is_supported_dtype(rhs.dtype)

  # Validate 'group_sizes'.
  if group_sizes.dtype != torch.int32:
    raise ValueError(
        f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")

  return lhs, group_sizes, common.select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
  return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
  tiles, rem = divmod(x, tx)
  if rem:
    tiles += 1
  return tiles, rem


GroupMetadata = Any  # TODO(enriqueps): Clean this up and use a namedtuple


def _make_group_metadata(
    *,
    group_sizes: 'jnp.ndarray',
    m: int,
    tm: int,
    start_group: 'jnp.ndarray',
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
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
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp

  num_groups = group_sizes.shape[0]
  end_group = start_group + num_nonzero_groups - 1

  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

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

  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

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

  if visit_empty_groups:
    # Insert one tile for empty groups.
    group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)
  group_ids = jnp.repeat(
      jnp.arange(num_groups, dtype=jnp.int32),
      group_tiles,
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
  partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0,
                                     group_sizes == 0)

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

  partial_tile_ids = jnp.where(partial_tile_mask, tiles_m,
                               group_offsets[:-1] // tm)

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] +
      1)

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
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles


def _zero_uninitialized_memory(
    out: 'jnp.ndarray',
    *,
    start_group: 'jnp.ndarray',
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> torch.Tensor:
  """Zero out uninitialized memory from output."""
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp

  group_offsets = group_metadata[0]
  group_start = group_offsets[start_group]
  group_end = group_offsets[start_group + num_nonzero_groups]
  valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
  valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
  return torch.from_numpy(np.array(jnp.where(valid_mask[:, None], out,
                                             0))).to('xla')


LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


def _gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    payload: str,
) -> torch.Tensor:
  import jax.numpy as jnp

  m, n = lhs.shape[0], rhs.shape[2]
  # Create the metadata we need for computation.
  group_sizes = jnp.asarray(group_sizes.numpy())
  group_metadata, num_active_tiles = _make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
      group_sizes=group_sizes,
      m=lhs.shape[0],
      tm=128,
      start_group=0,
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )
  group_metadata0 = torch.from_numpy(np.array(group_metadata[0])).to(
      torch.int32).to("xla")
  group_metadata1 = torch.from_numpy(np.array(group_metadata[1])).to("xla")
  group_metadata2 = torch.from_numpy(np.array(group_metadata[2])).to("xla")
  num_active_tiles = torch.tensor(np.array(num_active_tiles)).to("xla")
  group_offset_torch = torch.tensor([0], dtype=torch.int32).to("xla")
  output_shape = torch.Size([m, n])
  out = torch_xla._XLAC._xla_tpu_custom_call([
      num_active_tiles, group_metadata0, group_metadata1, group_metadata2,
      group_offset_torch, lhs, rhs
  ], payload, [output_shape], [lhs.dtype])
  return out


def gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor
) -> torch.Tensor:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, k, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.

  Returns:
    A 2d, jnp.ndarray with shape [m, n].
  """
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  from jax.experimental.pallas.ops.tpu.megablox import gmm
  from torch_xla.experimental.custom_kernel import trace_pallas

  payload, _ = trace_pallas(gmm, lhs, rhs, group_sizes)
  out = _gmm(lhs, rhs, group_sizes, payload)
  return out
