"""Grouped matrix multiplication kernels for TPU written in Pallas."""

from typing import Any, Callable, Optional, Union, Tuple
import common_pt as common
import torch
import torch_xla
import numpy as np
torch_xla._XLAC._init_computation_client()

import jax
import jax.numpy as jnp

def _validate_args(
    *,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    expected_rhs_dims: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.dtype]:
  """Validates the arguments for the gmm function."""
  # Validate 'lhs'.
  if lhs.dim() != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.dim()}-tensor.")
  common.assert_is_supported_dtype(lhs.dtype)

  # Validate 'rhs'.
  if rhs.dim() != expected_rhs_dims:
    raise ValueError(
        f"Expected {expected_rhs_dims}-tensor for 'rhs' but got"
        f" {rhs.dim()}-tensor."
    )
  common.assert_is_supported_dtype(rhs.dtype)

  # Validate 'group_sizes'.
  if group_sizes.dtype != torch.int32:
    raise ValueError(
        f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}."
    )

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


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
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
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
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
  partial_tile_mask = jnp.logical_or(
      (group_offsets[:-1] % tm) == 0, group_sizes == 0
  )

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

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
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles

def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> torch.Tensor:
  """Zero out uninitialized memory from output."""
  group_offsets = group_metadata[0]
  group_start = group_offsets[start_group]
  group_end = group_offsets[start_group + num_nonzero_groups]
  valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
  valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
  return torch.from_numpy(np.array(jnp.where(valid_mask[:, None], out, 0))).to('xla')

LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]

def gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    payload: str,
    preferred_element_type: torch.dtype = torch.float32,
    tiling: Optional[Union[tuple[int, int, int], LutFn]] = (128, 128, 128),
    group_offset: Optional[torch.Tensor] = None,
    existing_out: Optional[torch.Tensor] = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> torch.Tensor:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, k, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    payload: pallas payload extracted from the pallas code on JAX.
    preferred_element_type: jnp.dtype, the element type for the output matrix.
    tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
    group_offset: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    existing_out: Existing output to write to.
    transpose_rhs: True if the rhs needs to be transposed.
    interpret: Whether or not to run the kernel in interpret mode, helpful for
      testing and debugging.

  Returns:
    A 2d, torch.Tensor with shape [m, n].
  """

  if existing_out is not None:
    assert isinstance(existing_out, jax.Array)
    expected_dtype = existing_out.dtype
    if expected_dtype != preferred_element_type:
      raise ValueError(
          "Existing output dtype must match preferred_element_type."
      )
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if group_offset.shape:
      raise ValueError(
          f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
      )
    group_offset = group_offset[None]
  num_current_groups = rhs.shape[0]
  num_total_groups = group_sizes.shape[0]
  lhs, group_sizes, input_dtype = _validate_args(
      lhs=lhs, rhs=rhs, group_sizes=group_sizes
  )

  # Gather shape information.
  m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
  if transpose_rhs:
    n = rhs.shape[1]

  # If tiling is callable, look up the problem dimensions in the LUT. If no tuned
  # tile dimensions are available throw an error.
  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tk, tn = tiling
  tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
  tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
  del n_rem

  # Create the metadata we need for computation.
  group_sizes = jnp.asarray(group_sizes.numpy())
  group_metadata, num_active_tiles = make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
      group_sizes=group_sizes,
      m=m,
      tm=tm,
      start_group=group_offset[0],
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )
  group_metadata0 = torch.from_numpy(np.array(group_metadata[0])).to(torch.int32).to("xla")
  group_metadata1 = torch.from_numpy(np.array(group_metadata[1])).to("xla")
  group_metadata2 = torch.from_numpy(np.array(group_metadata[2])).to("xla")
  num_active_tiles = torch.tensor(np.array(num_active_tiles)).to("xla")
  group_offset_torch = torch.from_numpy(np.array(group_offset)).to("xla")

  # It returns the shape and type of tensors
  def shape_dtype(q, *arg):
    return [(q.shape, q.dtype)]

  output_shape_dtype = shape_dtype(lhs)
  output_shapes = [shape for shape, _ in output_shape_dtype]
  output_dtypes = [dtype for _, dtype in output_shape_dtype]
  print("output_shapes", output_shapes)
  print("output_dtypes", output_dtypes)
  print("group_metadata0.dtype", group_metadata0.dtype)
  print("group_metadata1.dtype", group_metadata1.dtype)
  print("group_metadata2.dtype", group_metadata2.dtype)
  print("lhs.dtype", lhs.dtype)
  print("rhs.dtype", rhs.dtype)
  print("group_offset_torch.dtype", group_offset_torch.dtype)
  print("num_active_tiles.dtype", num_active_tiles.dtype)

  print("group_metadata0.shape", group_metadata0.shape)
  print("group_metadata1.shape", group_metadata1.shape)
  print("group_metadata2.shape", group_metadata2.shape)
  print("group_offset_torch.shape", group_offset_torch.shape)
  print("num_active_tiles.shape", num_active_tiles.shape)
  # print("num_active_tiles", num_active_tiles)

  out = torch_xla._XLAC._xla_tpu_custom_call([num_active_tiles, group_metadata0, group_metadata1, group_metadata2, group_offset_torch, lhs, rhs], payload, output_shapes, output_dtypes)
  # print("PyTorch/XLA - gmm call output", out)

  if existing_out is None and num_current_groups < num_total_groups:
    print("Milad: in if block!!! does it work right?")
    out = _zero_uninitialized_memory(
        out,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        group_metadata=group_metadata,
    )
  return out

import numpy as np
def reference_gmm(
    lhs: np.array,
    rhs: np.array,
    group_sizes: np.array,
    preferred_element_type: np.dtype = np.float32,
) -> np.array:

  start = 0
  out = []
  for i, size in enumerate(group_sizes):
    result = np.dot(
        lhs[start : start + size, :],
        rhs[i, :, :]
    )

    out.append(result)
    start += group_sizes[i]
  return np.array(np.concatenate(out))

# import functools
# partial = functools.partial
# def reference_gmm_jax(
#     lhs: np.array,
#     rhs: np.array,
#     group_sizes: np.array,
#     preferred_element_type: np.dtype = np.float32,
# ) -> np.array:

#   import jax
#   import jax.numpy as jnp

#   lhs = jnp.asarray(lhs)
#   rhs = jnp.asarray(rhs)
#   group_sizes = jnp.asarray(group_sizes)

#   out, vjpfun = jax.vjp(
#     partial(
#         mblx.gmm,
#         preferred_element_type=out_dtype,
#         transpose_rhs=False,
#         interpret=False,
#     ),
#     lhs,
#     rhs,
#     group_sizes,
#  )

#   print("jax output")
#   print(out)

def group_sizes_strategy(
    m: int, num_groups: int
) -> torch.Tensor:
  # Randomly sample the ends of the groups in the m-dimension. Let the fuzzer
  # sample with replacement so that it's possible to get zero-sized groups. Get
  # 'num_groups - 1' run ends. The final group will end at 'm'.
  ends_no_final = np.sort(
      np.array(
          [
              np.random.randint(low=0, high=m)
              for _ in range(num_groups - 1)
          ],
          dtype=np.int32,
      ),
  )
  ends = np.concatenate([ends_no_final, np.array([m], dtype=np.int32)])

  # Calculate the run starts by shifting ends 1 to the right. The first run
  # starts at zero.
  starts = np.concatenate([np.zeros(1, dtype=np.int32), ends_no_final])
  return torch.from_numpy(ends - starts).to(torch.int32)

def tolerances(
    lhs_dtype: torch.dtype, rhs_dtype: torch.dtype, out_dtype: torch.dtype
) -> tuple[float, float]:
  if (
      lhs_dtype == torch.bfloat16
      or rhs_dtype == torch.bfloat16
      or out_dtype == torch.bfloat16
  ):
    return 1e-3, 1e-1  # atol, rtol
  return 1e-3, 1e-2  # atol, rtol

if __name__ == '__main__':
  payload_bf16_128 = "{\"custom_call_config\": {\"body\": \"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dWJpAGFyaXRoLm11bGkAdHB1LmlvdGEAYXJpdGguYW5kaQBhcml0aC5zZWxlY3QAL3Vzci9sb2NhbC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvbWVnYWJsb3gvZ21tLnB5AHZhbHVlAF9nZXRfc3RvcmVfbWFzawAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoMiwpLCAoKSldLCBbKl0pLCkpXQAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSkpKSwgKDEyOCwgMTI4KSwgKCkpXSwgW10pLCkpXQBmdW5jdGlvbl90eXBlAHN5bV9uYW1lAGtlcm5lbABwcmVkaWNhdGUAX2FjY3VtAF9zdG9yZV9hY2N1bQB0cmFuc2Zvcm1faW5kaWNlcwB3aW5kb3dfYm91bmRzAC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pLCBDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSkpLCAoMTI4LCAxMjgpLCAoKSldLCBbXSksKSldAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAC9lcQAvY29udmVydF9lbGVtZW50X3R5cGVbbmV3X2R0eXBlPWludDMyIHdlYWtfdHlwZT1GYWxzZV0ALQAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoMywpLCAoKSldLCBbKl0pLCkpXQBzdGFibGVfbW9zYWljLnZlcnNpb24AZGltZW5zaW9uX3NlbWFudGljcwBpdGVyYXRpb25fYm91bmRzAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAG1haW4Ad2luZG93X3BhcmFtcwBvdXRfdHJhbnNmb3JtX2luZGljZXMAZ21tADxtb2R1bGU+AC9yb290L21lZ2FibG94X2pheC5weQByaHNfdHJhbnNmb3JtX2luZGljZXMAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDEsKSwgKCkpXSwgWypdKSwpKV0Ab3ZlcmZsb3dGbGFncwAvc3ViAGxoc190cmFuc2Zvcm1faW5kaWNlcwAvY29uZFtsaW5lYXI9KEZhbHNlLCldAC9jb25kW2xpbmVhcj0oRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlKV0AL2RvdF9nZW5lcmFsW2RpbWVuc2lvbl9udW1iZXJzPSgoKDEsKSwgKDAsKSksICgoKSwgKCkpKSBwcmVjaXNpb249Tm9uZSBwcmVmZXJyZWRfZWxlbWVudF90eXBlPWZsb2F0MzJdAHRyYW5zcG9zZV9saHMAdHJhbnNwb3NlX3JocwBmYXN0bWF0aAAvbXVsAGRpbWVuc2lvbgAvaW90YVtkdHlwZT1pbnQzMiBzaGFwZT0oMTI4LCAxMjgpIGRpbWVuc2lvbj0wXQAvZ2UAL2x0AC9hbmQAL3NlbGVjdF9uAC9icm9hZGNhc3RfaW5fZGltW3NoYXBlPSgxMjgsIDEyOCkgYnJvYWRjYXN0X2RpbWVuc2lvbnM9KCldAF96ZXJvX2FjYwA=\", \"cost_estimate\": {\"flops\": 4194304, \"transcendentals\": 0, \"bytes_accessed\": 163840}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
  payload_f32_g1_128 = "{\"custom_call_config\": {\"body\": \"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aXRoLnNlbGVjdAAvdXNyL2xvY2FsL2xpYi9weXRob24zLjEwL3NpdGUtcGFja2FnZXMvamF4L2V4cGVyaW1lbnRhbC9wYWxsYXMvb3BzL3RwdS9tZWdhYmxveC9nbW0ucHkAdmFsdWUAX2dldF9zdG9yZV9tYXNrAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKCosKSksICgxLCksICgpKV0sIFsqXSksKSldAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pLCBDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSkpLCAoMTI4LCAxMjgpLCAoKSldLCBbXSksKSldAGZ1bmN0aW9uX3R5cGUAc3ltX25hbWUAa2VybmVsAHByZWRpY2F0ZQBfYWNjdW0AX3N0b3JlX2FjY3VtAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL2FkZAAvc3dhcFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSksIEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pKSksICgxMjgsIDEyOCksICgpKV0sIFtdKSwpKV0AdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAL2VxAC9jb252ZXJ0X2VsZW1lbnRfdHlwZVtuZXdfZHR5cGU9aW50MzIgd2Vha190eXBlPUZhbHNlXQAtAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKCosKSksICgyLCksICgpKV0sIFsqXSksKSldAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAG91dF90cmFuc2Zvcm1faW5kaWNlcwBnbW0APG1vZHVsZT4AL3Jvb3QvdGVzdC5weQByaHNfdHJhbnNmb3JtX2luZGljZXMAb3ZlcmZsb3dGbGFncwAvc3ViAGxoc190cmFuc2Zvcm1faW5kaWNlcwAvY29uZFtsaW5lYXI9KEZhbHNlLCldAC9jb25kW2xpbmVhcj0oRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlKV0AL2RvdF9nZW5lcmFsW2RpbWVuc2lvbl9udW1iZXJzPSgoKDEsKSwgKDAsKSksICgoKSwgKCkpKSBwcmVjaXNpb249Tm9uZSBwcmVmZXJyZWRfZWxlbWVudF90eXBlPWZsb2F0MzJdAHRyYW5zcG9zZV9saHMAdHJhbnNwb3NlX3JocwBmYXN0bWF0aAAvbXVsAGRpbWVuc2lvbgAvaW90YVtkdHlwZT1pbnQzMiBzaGFwZT0oMTI4LCAxMjgpIGRpbWVuc2lvbj0wXQAvZ2UAL2x0AC9hbmQAL3NlbGVjdF9uAC9icm9hZGNhc3RfaW5fZGltW3NoYXBlPSgxMjgsIDEyOCkgYnJvYWRjYXN0X2RpbWVuc2lvbnM9KCldAF96ZXJvX2FjYwA=\", \"cost_estimate\": {\"flops\": 4194304, \"transcendentals\": 0, \"bytes_accessed\": 196608}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
  payload_f32_g2_128 = "{\"custom_call_config\": {\"body\": \"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b3RhAGFyaXRoLmFuZGkAYXJpdGguc2VsZWN0AC91c3IvbG9jYWwvbGliL3B5dGhvbjMuMTAvc2l0ZS1wYWNrYWdlcy9qYXgvZXhwZXJpbWVudGFsL3BhbGxhcy9vcHMvdHB1L21lZ2FibG94L2dtbS5weQB2YWx1ZQBfZ2V0X3N0b3JlX21hc2sAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDIsKSwgKCkpXSwgWypdKSwpKV0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSksIEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pKSksICgxMjgsIDEyOCksICgpKV0sIFtdKSwpKV0AZnVuY3Rpb25fdHlwZQBzeW1fbmFtZQBrZXJuZWwAcHJlZGljYXRlAF9hY2N1bQBfc3RvcmVfYWNjdW0AdHJhbnNmb3JtX2luZGljZXMAd2luZG93X2JvdW5kcwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSkpKSwgKDEyOCwgMTI4KSwgKCkpXSwgW10pLCkpXQB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgAvZXEAL2NvbnZlcnRfZWxlbWVudF90eXBlW25ld19kdHlwZT1pbnQzMiB3ZWFrX3R5cGU9RmFsc2VdAC0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDMsKSwgKCkpXSwgWypdKSwpKV0Ac3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAb3V0X3RyYW5zZm9ybV9pbmRpY2VzAGdtbQA8bW9kdWxlPgAvcm9vdC90ZXN0LnB5AHJoc190cmFuc2Zvcm1faW5kaWNlcwAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoMSwpLCAoKSldLCBbKl0pLCkpXQBvdmVyZmxvd0ZsYWdzAC9zdWIAbGhzX3RyYW5zZm9ybV9pbmRpY2VzAC9jb25kW2xpbmVhcj0oRmFsc2UsKV0AL2NvbmRbbGluZWFyPShGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UpXQAvZG90X2dlbmVyYWxbZGltZW5zaW9uX251bWJlcnM9KCgoMSwpLCAoMCwpKSwgKCgpLCAoKSkpIHByZWNpc2lvbj1Ob25lIHByZWZlcnJlZF9lbGVtZW50X3R5cGU9ZmxvYXQzMl0AdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9tdWwAZGltZW5zaW9uAC9pb3RhW2R0eXBlPWludDMyIHNoYXBlPSgxMjgsIDEyOCkgZGltZW5zaW9uPTBdAC9nZQAvbHQAL2FuZAAvc2VsZWN0X24AL2Jyb2FkY2FzdF9pbl9kaW1bc2hhcGU9KDEyOCwgMTI4KSBicm9hZGNhc3RfZGltZW5zaW9ucz0oKV0AX3plcm9fYWNjAA==\", \"cost_estimate\": {\"flops\": 4194304, \"transcendentals\": 0, \"bytes_accessed\": 262144}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
  payload_f32_g8_128 = "{\"custom_call_config\": {\"body\": \"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b3RhAGFyaXRoLmFuZGkAYXJpdGguc2VsZWN0AC91c3IvbG9jYWwvbGliL3B5dGhvbjMuMTAvc2l0ZS1wYWNrYWdlcy9qYXgvZXhwZXJpbWVudGFsL3BhbGxhcy9vcHMvdHB1L21lZ2FibG94L2dtbS5weQB2YWx1ZQBfZ2V0X3N0b3JlX21hc2sAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDgsKSwgKCkpXSwgWypdKSwpKV0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSksIEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pKSksICgxMjgsIDEyOCksICgpKV0sIFtdKSwpKV0AZnVuY3Rpb25fdHlwZQBzeW1fbmFtZQBrZXJuZWwAcHJlZGljYXRlAF9hY2N1bQBfc3RvcmVfYWNjdW0AdHJhbnNmb3JtX2luZGljZXMAd2luZG93X2JvdW5kcwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSkpKSwgKDEyOCwgMTI4KSwgKCkpXSwgW10pLCkpXQB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgAvZXEAL2NvbnZlcnRfZWxlbWVudF90eXBlW25ld19kdHlwZT1pbnQzMiB3ZWFrX3R5cGU9RmFsc2VdAC0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDksKSwgKCkpXSwgWypdKSwpKV0Ac3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAb3V0X3RyYW5zZm9ybV9pbmRpY2VzAGdtbQA8bW9kdWxlPgAvcm9vdC90ZXN0LnB5AHJoc190cmFuc2Zvcm1faW5kaWNlcwAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoMSwpLCAoKSldLCBbKl0pLCkpXQBvdmVyZmxvd0ZsYWdzAC9zdWIAbGhzX3RyYW5zZm9ybV9pbmRpY2VzAC9jb25kW2xpbmVhcj0oRmFsc2UsKV0AL2NvbmRbbGluZWFyPShGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UpXQAvZG90X2dlbmVyYWxbZGltZW5zaW9uX251bWJlcnM9KCgoMSwpLCAoMCwpKSwgKCgpLCAoKSkpIHByZWNpc2lvbj1Ob25lIHByZWZlcnJlZF9lbGVtZW50X3R5cGU9ZmxvYXQzMl0AdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9tdWwAZGltZW5zaW9uAC9pb3RhW2R0eXBlPWludDMyIHNoYXBlPSgxMjgsIDEyOCkgZGltZW5zaW9uPTBdAC9nZQAvbHQAL2FuZAAvc2VsZWN0X24AL2Jyb2FkY2FzdF9pbl9kaW1bc2hhcGU9KDEyOCwgMTI4KSBicm9hZGNhc3RfZGltZW5zaW9ucz0oKV0AX3plcm9fYWNjAA==\", \"cost_estimate\": {\"flops\": 4194304, \"transcendentals\": 0, \"bytes_accessed\": 655360}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
  payload_bf16_g8_128 = "{\"custom_call_config\": {\"body\": \"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dWJpAGFyaXRoLm11bGkAdHB1LmlvdGEAYXJpdGguYW5kaQBhcml0aC5zZWxlY3QAL3Vzci9sb2NhbC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvbWVnYWJsb3gvZ21tLnB5AHZhbHVlAF9nZXRfc3RvcmVfbWFzawAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoOCwpLCAoKSldLCBbKl0pLCkpXQAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSkpKSwgKDEyOCwgMTI4KSwgKCkpXSwgW10pLCkpXQBmdW5jdGlvbl90eXBlAHN5bV9uYW1lAGtlcm5lbABwcmVkaWNhdGUAX2FjY3VtAF9zdG9yZV9hY2N1bQB0cmFuc2Zvcm1faW5kaWNlcwB3aW5kb3dfYm91bmRzAC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pLCBDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSkpLCAoMTI4LCAxMjgpLCAoKSldLCBbXSksKSldAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAC9lcQAvY29udmVydF9lbGVtZW50X3R5cGVbbmV3X2R0eXBlPWludDMyIHdlYWtfdHlwZT1GYWxzZV0ALQAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoOSwpLCAoKSldLCBbKl0pLCkpXQBzdGFibGVfbW9zYWljLnZlcnNpb24AZGltZW5zaW9uX3NlbWFudGljcwBpdGVyYXRpb25fYm91bmRzAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAG1haW4Ad2luZG93X3BhcmFtcwBvdXRfdHJhbnNmb3JtX2luZGljZXMAZ21tADxtb2R1bGU+AC9yb290L3Rlc3QucHkAcmhzX3RyYW5zZm9ybV9pbmRpY2VzAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKCosKSksICgxLCksICgpKV0sIFsqXSksKSldAG92ZXJmbG93RmxhZ3MAL3N1YgBsaHNfdHJhbnNmb3JtX2luZGljZXMAL2NvbmRbbGluZWFyPShGYWxzZSwpXQAvY29uZFtsaW5lYXI9KEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSldAC9kb3RfZ2VuZXJhbFtkaW1lbnNpb25fbnVtYmVycz0oKCgxLCksICgwLCkpLCAoKCksICgpKSkgcHJlY2lzaW9uPU5vbmUgcHJlZmVycmVkX2VsZW1lbnRfdHlwZT1mbG9hdDMyXQB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAZmFzdG1hdGgAL211bABkaW1lbnNpb24AL2lvdGFbZHR5cGU9aW50MzIgc2hhcGU9KDEyOCwgMTI4KSBkaW1lbnNpb249MF0AL2dlAC9sdAAvYW5kAC9zZWxlY3RfbgAvYnJvYWRjYXN0X2luX2RpbVtzaGFwZT0oMTI4LCAxMjgpIGJyb2FkY2FzdF9kaW1lbnNpb25zPSgpXQBfemVyb19hY2MA\", \"cost_estimate\": {\"flops\": 4194304, \"transcendentals\": 0, \"bytes_accessed\": 360448}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
  payload_bf16_g1_128 = "{\"custom_call_config\": {\"body\": \"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dS5pb3RhAGFyaXRoLmFuZGkAYXJpdGguc2VsZWN0AC91c3IvbG9jYWwvbGliL3B5dGhvbjMuMTAvc2l0ZS1wYWNrYWdlcy9qYXgvZXhwZXJpbWVudGFsL3BhbGxhcy9vcHMvdHB1L21lZ2FibG94L2dtbS5weQB2YWx1ZQBfZ2V0X3N0b3JlX21hc2sAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDEsKSwgKCkpXSwgWypdKSwpKV0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSksIEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pKSksICgxMjgsIDEyOCksICgpKV0sIFtdKSwpKV0AZnVuY3Rpb25fdHlwZQBzeW1fbmFtZQBrZXJuZWwAcHJlZGljYXRlAF9hY2N1bQBfc3RvcmVfYWNjdW0AdHJhbnNmb3JtX2luZGljZXMAd2luZG93X2JvdW5kcwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSkpKSwgKDEyOCwgMTI4KSwgKCkpXSwgW10pLCkpXQB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgAvZXEAL2NvbnZlcnRfZWxlbWVudF90eXBlW25ld19kdHlwZT1pbnQzMiB3ZWFrX3R5cGU9RmFsc2VdAC0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDIsKSwgKCkpXSwgWypdKSwpKV0Ac3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAb3V0X3RyYW5zZm9ybV9pbmRpY2VzAGdtbQA8bW9kdWxlPgAvcm9vdC90ZXN0LnB5AHJoc190cmFuc2Zvcm1faW5kaWNlcwBvdmVyZmxvd0ZsYWdzAC9zdWIAbGhzX3RyYW5zZm9ybV9pbmRpY2VzAC9jb25kW2xpbmVhcj0oRmFsc2UsKV0AL2NvbmRbbGluZWFyPShGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UpXQAvZG90X2dlbmVyYWxbZGltZW5zaW9uX251bWJlcnM9KCgoMSwpLCAoMCwpKSwgKCgpLCAoKSkpIHByZWNpc2lvbj1Ob25lIHByZWZlcnJlZF9lbGVtZW50X3R5cGU9ZmxvYXQzMl0AdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9tdWwAZGltZW5zaW9uAC9pb3RhW2R0eXBlPWludDMyIHNoYXBlPSgxMjgsIDEyOCkgZGltZW5zaW9uPTBdAC9nZQAvbHQAL2FuZAAvc2VsZWN0X24AL2Jyb2FkY2FzdF9pbl9kaW1bc2hhcGU9KDEyOCwgMTI4KSBicm9hZGNhc3RfZGltZW5zaW9ucz0oKV0AX3plcm9fYWNjAA==\", \"cost_estimate\": {\"flops\": 4194304, \"transcendentals\": 0, \"bytes_accessed\": 131072}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"
  payload_f32_g8_512 = "{\"custom_call_config\": {\"body\": \"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Lm11bGkAdHB1LmlvdGEAYXJpdGguYW5kaQBhcml0aC5zZWxlY3QAL3Vzci9sb2NhbC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvbWVnYWJsb3gvZ21tLnB5AHZhbHVlAF9nZXRfc3RvcmVfbWFzawAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoMTEsKSwgKCkpXSwgWypdKSwpKV0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSksIEN1c3RvbU5vZGUoU2xpY2VbKDAsIDEyOCwgMSldLCBbTm9uZSwgTm9uZV0pKSksICgxMjgsIDEyOCksICgpKV0sIFtdKSwpKV0AZnVuY3Rpb25fdHlwZQBzeW1fbmFtZQBrZXJuZWwAcHJlZGljYXRlAF9hY2N1bQBfc3RvcmVfYWNjdW0AdHJhbnNmb3JtX2luZGljZXMAd2luZG93X2JvdW5kcwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCAxMjgsIDEpXSwgW05vbmUsIE5vbmVdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4LCAxKV0sIFtOb25lLCBOb25lXSkpKSwgKDEyOCwgMTI4KSwgKCkpXSwgW10pLCkpXQB0cmFuc2Zvcm1fMAB0cmFuc2Zvcm1fMQB0cmFuc2Zvcm1fMgAvZXEAL2NvbnZlcnRfZWxlbWVudF90eXBlW25ld19kdHlwZT1pbnQzMiB3ZWFrX3R5cGU9RmFsc2VdAC0AL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwpKSwgKDksKSwgKCkpXSwgWypdKSwpKV0Ac3RhYmxlX21vc2FpYy52ZXJzaW9uAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAb3V0X3RyYW5zZm9ybV9pbmRpY2VzAGdtbQA8bW9kdWxlPgAvcm9vdC90ZXN0LnB5AHJoc190cmFuc2Zvcm1faW5kaWNlcwAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKCgqLCkpLCAoMSwpLCAoKSldLCBbKl0pLCkpXQBvdmVyZmxvd0ZsYWdzAC9zdWIAbGhzX3RyYW5zZm9ybV9pbmRpY2VzAC9jb25kW2xpbmVhcj0oRmFsc2UsKV0AL2NvbmRbbGluZWFyPShGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UsIEZhbHNlLCBGYWxzZSwgRmFsc2UpXQAvZG90X2dlbmVyYWxbZGltZW5zaW9uX251bWJlcnM9KCgoMSwpLCAoMCwpKSwgKCgpLCAoKSkpIHByZWNpc2lvbj1Ob25lIHByZWZlcnJlZF9lbGVtZW50X3R5cGU9ZmxvYXQzMl0AdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9tdWwAZGltZW5zaW9uAC9pb3RhW2R0eXBlPWludDMyIHNoYXBlPSgxMjgsIDEyOCkgZGltZW5zaW9uPTBdAC9nZQAvbHQAL2FuZAAvc2VsZWN0X24AL2Jyb2FkY2FzdF9pbl9kaW1bc2hhcGU9KDEyOCwgMTI4KSBicm9hZGNhc3RfZGltZW5zaW9ucz0oKV0AX3plcm9fYWNjAA==\", \"cost_estimate\": {\"flops\": 268435456, \"transcendentals\": 0, \"bytes_accessed\": 16777216}, \"serialization_format\": 1, \"needs_layout_passes\": true}, \"implicit_sharding\": {\"type\": \"MANUAL\"}}"

  seed = 421
  k = m = n = 512
  num_groups = 8

  lhs_dtype = rhs_dtype = out_dtype = torch.float32

  lhs = torch.rand(m, k, dtype=lhs_dtype).to('xla')
  rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype).to('xla')

  group_sizes = group_sizes_strategy(m=m, num_groups=num_groups)

  out = gmm(lhs, rhs, group_sizes, payload_f32_g8_512)
  print("PyTocrh/XLA Result", out)
  
  ref_out = reference_gmm(lhs.to('cpu').float().numpy(), rhs.to('cpu').float().numpy(), group_sizes.numpy())
  print("Reference Result", ref_out)
  
  atol, rtol = tolerances(lhs_dtype, rhs_dtype, out_dtype)
  print(np.allclose(np.array(ref_out, dtype=float), np.array(out[0].to('cpu').float(), dtype=float), rtol=rtol, atol=atol))