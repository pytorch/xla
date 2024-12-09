import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla._internal.utils as _utils
from torch_xla.distributed.spmd import XLAShardedTensor, XLAShard
import torch_xla.runtime as xr

import numpy as np
import functools
import itertools
from typing import Tuple, Union, List, Sequence, Any, Optional, Set
from enum import IntEnum

from torch.amp import custom_fwd, custom_bwd


class Mesh:
  """Describe the logical XLA device topology mesh and the underlying resources.

  Args:
    device_ids (Union[np.ndarray, List]): A raveled list of devices (IDs) in a custom order. The list is reshaped
        to an `mesh_shape` array, filling the elements using C-like index order.

    mesh_shape (Tuple[int, ...]): A int tuple describing the logical topology shape
        of the device mesh, and each element describes the number of devices in
        the corresponding axis.

    axis_names (Tuple[str, ...]): A sequence of resource axis names to be assigned to the dimensions
        of the `devices` argument. Its length should match the rank of `devices`.

  Example:

    >>> mesh_shape = (4, 2)
    >>> num_devices = len(xm.get_xla_supported_devices())
    >>> device_ids = np.array(range(num_devices))
    >>> mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
    >>> mesh.get_logical_mesh()
    >>> array([[0, 1],
              [2, 3],
              [4, 5],
              [6, 7]])
    >>> mesh.shape()
    OrderedDict([('x', 4), ('y', 2)])
  """

  device_ids: np.ndarray
  mesh_shape: Tuple[int, ...]
  axis_names: Tuple[str, ...]

  def __init__(self,
               device_ids: Union[np.ndarray, List],
               mesh_shape: Tuple[int, ...],
               axis_names: Tuple[str, ...] = None):
    if not isinstance(device_ids, np.ndarray):
      device_ids = np.array(device_ids)
    assert (axis_names is None) or (len(mesh_shape) == len(axis_names))
    assert axis_names is None or (len(set(axis_names)) == len(axis_names))
    assert (len(device_ids) == np.prod(mesh_shape))
    assert len(device_ids) == len(np.unique(device_ids))
    self.device_ids = device_ids
    self.mesh_shape = mesh_shape
    self.axis_names = axis_names
    assert all(d < self.size() for d in device_ids)

  def size(self):
    return np.prod(self.mesh_shape)

  def shape(self):
    if self.axis_names is None:
      return OrderedDict(
          (dim, size) for dim, size in enumerate(self.mesh_shape))
    return OrderedDict(
        (name, size) for name, size in zip(self.axis_names, self.mesh_shape))

  def get_logical_mesh(self):
    return self.device_ids.reshape(self.mesh_shape)

  def get_axis_name_idx(self, name: str) -> int:
    if name not in self.axis_names:
      return None
    return self.axis_names.index(name)

  @functools.lru_cache(maxsize=None)
  def _get_op_sharding_args(self, partition_spec: Tuple):
    partition_spec = _translate_named_partition_spec(self, partition_spec)
    flat_specs = np.hstack([d for d in partition_spec])
    specs = [d for d in flat_specs if d is not None]
    assert all(d >= 0 and d < len(self.mesh_shape) for d in specs), \
      f"partition_spec ({partition_spec}) contains out of bound index into mesh_shape."
    assert len(specs) == len(np.unique(specs)), \
    f"Each device mesh dimension should appear at most once in partition_spec {partition_spec}."

    tile_assignment = _get_tile_assignment(self, partition_spec)
    if len(tile_assignment.shape) > len(partition_spec):
      # Use partial replication for sharding a tensor over a higher-rank mesh
      sharding_type = ShardingType.PARTIAL
    else:
      sharding_type = _get_sharding_type(partition_spec, self.size())
    replicate_dims = {i for i, d in enumerate(partition_spec) if d is None}
    group_assignment, replication_groups = _get_group_assignment(
        sharding_type, tile_assignment, len(partition_spec), replicate_dims)

    tile_assignment = tile_assignment.tolist()
    sharding_type = int(sharding_type)
    return tile_assignment, group_assignment, replication_groups, sharding_type

  @functools.lru_cache(maxsize=None)
  def get_op_sharding(self,
                      partition_spec: Tuple) -> torch_xla._XLAC.OpSharding:
    """
    Return the OpSharding for the given partition spec. This is an expensive
    operation as the mesh grows, so the value is cached for reuse.
    """
    # For scalar tensors, it can only be replicated.
    # We have made sure len(t.shape) == len(partition_spec)
    # in mark_sharding API.
    if len(partition_spec) == 0:
      return torch_xla._XLAC.OpSharding([], [], [], ShardingType.REPLICATED)

    tile_assignment, group_assignment, replication_groups, sharding_type = self._get_op_sharding_args(
        partition_spec)
    return torch_xla._XLAC.OpSharding(tile_assignment, group_assignment,
                                      replication_groups, sharding_type)


_GLOBAL_MESH: Mesh = None


def set_global_mesh(mesh: Mesh):
  """
  Set the global mesh that can be used for the current process.

  Args:
    mesh: (Mesh) Mesh object that will be the global mesh.

  Example:

    >>> import torch_xla.distributed.spmd as xs
    >>> mesh = xs.get_1d_mesh("data")
    >>> xs.set_global_mesh(mesh)
  """
  global _GLOBAL_MESH
  _GLOBAL_MESH = mesh


def get_global_mesh() -> Optional[Mesh]:
  """
  Get the global mesh for the current process.

  Returns:
    mesh: (Optional[Mesh]) Mesh object if global mesh is set, otherwise return None.

  Example:

    >>> import torch_xla.distributed.spmd as xs
    >>> xs.get_global_mesh()
  """
  global _GLOBAL_MESH
  return _GLOBAL_MESH


def get_1d_mesh(axis_name: Optional[str] = None) -> Mesh:
  """
  Helper function to return the mesh with all devices in one dimension.

  Args:
    axis_name: (Optional[str]) optional string to represent the axis name of the mesh

  Returns:
    Mesh: Mesh object

  Example:

    >>> # This example is assuming 1 TPU v4-8
    >>> import torch_xla.distributed.spmd as xs
    >>> mesh = xs.get_1d_mesh("data")
    >>> print(mesh.mesh_shape)
    (4,)
    >>> print(mesh.axis_names)
    ('data',)
  """
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (num_devices,)
  device_ids = np.array(range(num_devices))
  if axis_name == None:
    return Mesh(device_ids, mesh_shape)
  else:
    return Mesh(device_ids, mesh_shape, (axis_name,))


# HybridDevice class has been inspired from jax's mesh_utils: https://github.com/google/jax/blob/fc5960f2b8b7a0ef74dbae4e27c5c08ff1564cff/jax/experimental/mesh_utils.py#L4ƒ
class HybridMesh(Mesh):
  """Creates a hybrid device mesh of devices connected with ICI and DCN networks.
    The shape of logical mesh should be ordered by increasing network-intensity
    e.g. [replica, data, model] where mdl has the most network communication
    requirements.

  Args:
    ici_mesh_shape: shape of the logical mesh for inner connected devices.
    dcn_mesh_shape: shape of logical mesh for outer connected devices.

  Example:

    >>> # This example is assuming 2 slices of v4-8.
    >>> ici_mesh_shape = (1, 4, 1) # (data, fsdp, tensor)
    >>> dcn_mesh_shape = (2, 1, 1)
    >>> mesh = HybridMesh(ici_mesh_shape, dcn_mesh_shape, ('data','fsdp','tensor'))
    >>> print(mesh.shape())
    >>> >> OrderedDict([('data', 2), ('fsdp', 4), ('tensor', 1)])
  """
  ici_mesh_shape: Tuple[int, ...]
  dcn_mesh_shape: Tuple[int, ...]

  def __init__(self,
               *,
               ici_mesh_shape: Tuple[int, ...],
               dcn_mesh_shape: Tuple[int, ...] = None,
               axis_names: Tuple[str, ...] = None):
    if dcn_mesh_shape == None:
      dcn_mesh_shape = tuple([1] * len(ici_mesh_shape))
    assert len(ici_mesh_shape) == len(dcn_mesh_shape)
    mesh_shape = tuple([x * y for x, y in zip(ici_mesh_shape, dcn_mesh_shape)])
    self.device_attributes = xr.global_runtime_device_attributes()
    self.device_attributes.sort(
        key=lambda attr: _utils.parse_xla_device(attr['name'])[1])

    if 'slice_index' in self.device_attributes[0] and np.prod(
        dcn_mesh_shape) == 1:
      raise ValueError('Provide dcn_mesh_shape to create a mesh for multislice')
    if 'slice_index' not in self.device_attributes[0] and np.prod(
        dcn_mesh_shape) > 1:
      raise ValueError('Invalid dcn_mesh_shape for single slice mesh')
    self.ici_mesh_shape = ici_mesh_shape
    self.dcn_mesh_shape = dcn_mesh_shape
    if np.prod(dcn_mesh_shape) > 1 and 'slice_index' in self.device_attributes[
        0]:  # multislice
      mesh = self._create_hybrid_device_mesh(self.ici_mesh_shape,
                                             self.dcn_mesh_shape)
    else:
      mesh = self._create_device_mesh(self.ici_mesh_shape)
    device_ids = mesh.flatten()
    super().__init__(device_ids, mesh_shape, axis_names)

  # This is imported from JAX: https://github.com/google/jax/blob/main/jax/experimental/mesh_utils.py#L172
  def _get_physical_tpu_mesh(self, devices: Sequence[int]) -> np.ndarray:
    r"""Rearrange TPU devices in a slice into a physical mesh.

      Args:
        devices: A list of device logical ordinals in a TPU slice.

      Returns:
        A np.ndarray of device logical ordinals with shape [global_x, global_y, global_z]. On
          v2 and v3, global_z is instead cores_per_chip (i.e., 2).
    """
    assert xm.xla_device_hw(xm.xla_device()) == 'TPU'
    # coords is a 3-dims tuple representing the device in physical mesh
    device_coords = [self.device_attributes[d]['coords'] for d in devices]
    dims = tuple(d + 1 for d in max(device_coords))
    out = np.empty(dims, dtype=int)
    for coords, d in zip(device_coords, devices):
      out[coords[0], coords[1], coords[2]] = d
    return out

  # This is imported from JAX: https://github.com/google/jax/blob/main/jax/experimental/mesh_utils.py#L64.
  def _create_device_mesh_for_nd_torus(
      self, physical_mesh: np.ndarray,
      mesh_shape: Sequence[int]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Assigns logical parallelism axes to physical axes of an N-D torus network.

      Given logical parallelism axes with sizes in `mesh_shape` and devices in an
      N-dimensional torus network represented by `physical_mesh`, maps each logical
      axis to one or more physical axes. Prefer to map more-performance-sensitive
      logical axes to larger numbers of physical axes to maximize the bandwidth
      available to them. Also prefer to assign logical axes to multiple physical
      axes of the same size (e.g., a 2D square) rather than multiple physical axes
      of different sizes when possible.

      Note that this routine will never split a physical axis over more than one
      logical axis (which would reduce total usable bandwidth but may sometimes be
      desired anyway). As a result, it will error out in cases where this is
      necessary to produce a valid mapping.

      Let's use a concrete example to explain the concepts and considerations.

      As an example, suppose the logical mesh is [data, model], for data and model
      parallelism respectively. Also suppose that data parallelism is less
      performance sensitive than model parallelism. Consider a 3D TPU pod slice of
      shape 4x4x16, represented by a physical mesh of shape (4, 4, 16).

      A TPU pod slice has equal bandwidth along all axes with wraparound links, but
      a 2D plane of size 4x4 may have faster XLA collective implementations than a
      non-square plane or a 1D subgroup. If the mesh_shape is [16, 16], we may want
      the more performance sensitive `model` axis to be mapped to the 4x4 XY plane.

      Args:
        physical_mesh: a np.ndarray of devices in the shape of the N-D torus
          physical topology.
        mesh_shape: shape of the logical mesh (size of the various logical
          parallelism axes), with axes ordered by increasing network intensity.

      Returns:
        An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
          each logical parallelism axis mapped to one or more physical mesh axes.
        The axis assignment (a list of length num_logical_axes, whose elements
          are tuples representing physical axis indices).
    """
    # Remaining physical axes to be assigned to logical axes.
    assignable_physical_mesh = list(physical_mesh.shape)
    # Map each logical axis to a subset of physical axes.
    assignment: List[Tuple[int, ...]] = [() for _ in mesh_shape]
    # Assign logical axes from highest network intensity to lowest.
    # `mesh_shape` is assumed to ordered by lowest network intensity first, so
    # reverse it first.
    # Assigns devices to 2D or 3D logical mesh.
    for logical_axis_index, logical_axis_size in reversed(
        list(enumerate(mesh_shape))):
      for num_axes in range(3, 0, -1):
        # map a combination of devices in physical axes to the logical axis.
        axes = itertools.combinations(assignable_physical_mesh, num_axes)
        indices = itertools.combinations(
            range(len(assignable_physical_mesh)), num_axes)
        for c_axes, c_indices in zip(axes, indices):
          if np.prod(c_axes) == logical_axis_size:
            assignment[logical_axis_index] = c_indices
            # Zero the assigned physical axes.
            assignable_physical_mesh = [
                0 if i in c_indices else v
                for i, v in enumerate(assignable_physical_mesh)
            ]
            break
        if assignment[logical_axis_index]:
          # We already found an assignment from one candidate above.
          break
      else:
        # If the num_axes for loop did not break, i.e. none of the candidates work
        # goto here with this while-else construct.
        if logical_axis_size > 1:
          raise NotImplementedError(
              'Failed to find assignment for logical_axis_index'
              f' {logical_axis_index} of size {logical_axis_size} with remaining'
              f' assignable mesh {assignable_physical_mesh}. The size of each'
              ' axis in your logical mesh must be equal to the product of'
              ' some subset of the physical mesh axis sizes. E.g logical mesh (4,'
              ' 16) is compatible with physical mesh 4x4x4 since 4=4 and 16=4x4.'
          )
    # Flatten the assignment
    transpose: List[int] = []
    for x in assignment:
      for y in x:
        transpose.append(int(y))
    return physical_mesh.transpose(transpose).reshape(mesh_shape), assignment

  def _create_device_mesh(self,
                          mesh_shape: Sequence[int],
                          devices: Sequence[Any] = None) -> Sequence[int]:
    """Creates a performant device mesh.

      Args:
        mesh_shape: shape of logical mesh, ordered by increasing network-intensity
          e.g. [replica, data, mdl] where mdl has the most network communication
          requirements.
        devices: optionally, the devices to construct a mesh for.

      Returns:
        A np.ndarray of devices with mesh_shape as its shape.
    """

    if devices is None:
      devices = np.arange(xr.global_runtime_device_count())
    if np.prod(mesh_shape) != len(devices):
      raise ValueError(
          f'Number of devices {len(devices)} must equal the product '
          f'of mesh_shape {mesh_shape}')
    physical_mesh = self._get_physical_tpu_mesh(devices)
    device_mesh, assignment = self._create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape)
    return device_mesh

  # This is imported from JAX: https://github.com/google/jax/blob/main/jax/experimental/mesh_utils.py#L288.
  def _create_hybrid_device_mesh(
      self, ici_mesh_shape: Sequence[int],
      dcn_mesh_shape: Sequence[int]) -> Sequence[int]:
    """Creates a device mesh for hybrid (e.g., ICI and DCN) parallelism.

      Args:
        ici_mesh_shape: shape of the logical mesh for the faster/inner network, ordered
          by increasing network intensity, e.g. [replica, data, mdl] where mdl has
          the most network communication requirements.
        dcn_mesh_shape: shape of the logical mesh for the slower/outer network,
          in the same order as mesh_shape.

      Returns:
        A np.ndarray of device logical ordinal with ici_mesh_shape * dcn_mesh_shape as its shape
        that can be fed into HybridMesh for hybrid parallelism.
    """
    granule_dict = defaultdict(list)
    for d, dev in enumerate(self.device_attributes):
      granule_dict[dev['slice_index']].append(d)
    # sorts devices based on slice_index.
    granules = list(granule_dict[key] for key in sorted(granule_dict.keys()))
    if np.prod(dcn_mesh_shape) != len(granules):
      raise ValueError(
          f'Number of slices {len(granules)} must equal the product of '
          f'dcn_mesh_shape {dcn_mesh_shape}')
    # creates a seperate internal mesh for each slice.
    per_granule_meshes = [
        self._create_device_mesh(ici_mesh_shape, granule)
        for granule in granules
    ]
    granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
    blocks = np.vectorize(
        lambda i: per_granule_meshes[i], otypes=[object])(
            granule_mesh)
    device_mesh = np.block(blocks.tolist())
    return device_mesh


class ShardingType(IntEnum):
  # ShardingType enum ID maps to OpSharidng.Type (https://shorturl.at/pvAJX)
  REPLICATED = 0
  MAXIMAL = 1
  TUPLE = 2
  TILED = 3
  MANUAL = 4
  PARTIAL = 5
  UNKNOWN = 6  # implicit replication. TODO(yeounoh) wait for auto-sharding support


def _get_sharding_type(partition_spec: Tuple[Union[int, None]],
                       num_devices: int) -> ShardingType:
  sharding_type = ShardingType.TILED
  if num_devices == 1:
    sharding_type = ShardingType.MAXIMAL
  elif all(d is None for d in partition_spec):
    sharding_type = ShardingType.REPLICATED
  elif any(d is None for d in partition_spec):
    sharding_type = ShardingType.PARTIAL
  return sharding_type


def _get_tile_assignment(
    mesh: Mesh, partition_spec: Tuple[Union[Tuple[int], int,
                                            None]]) -> np.ndarray:
  """
  Permute the given mesh to create the tile assignment based on the partition
  spec. Returns the tiling assignment as a numpy ndarray.

  If the input partition_spec combines multiple logical mesh axes over a single
  tensor axis, the resulting tiling assignment will combine the specified axes
  into a single axis.
  """
  # Flatten the partition spec and ensure that it is fully specified over the
  # mesh for permutation.
  tiled_dims = [x for x in partition_spec if x is not None]
  permutation = np.hstack(tiled_dims).tolist() if tiled_dims else []
  missing_axes = sorted(set(range(len(mesh.shape()))) - set(permutation))
  tile_assignment = mesh.get_logical_mesh().transpose(permutation +
                                                      missing_axes)

  # For any tuples in the partition_spec, the grouped axes will be adjacent
  # after the permutation. Combine these dimensions into a single axis.
  for i, spec in enumerate(tiled_dims):
    if isinstance(spec, tuple):
      shape = tile_assignment.shape
      tile_assignment = tile_assignment.reshape(shape[:i] + (-1,) +
                                                shape[i + len(spec):])

  return tile_assignment


# Produce group assignment for partial replication. Partial replication tiles
# groups (a.k.a. sub-groups) where the shards are fully replicated within each
# sub-group. `replication_groups` is a list of groups as lists, where each group
# contains the participating device IDs. `group_assignment` describes the group
# placement and the overall mesh, where each element is the group ID.
# The tile_assignment should be the result of `_get_tile_assignment` so that all
# tiled dimensions are in the first axes and replicated dimensions are in the
# remaining axes.
def _get_group_assignment(sharding_type: ShardingType,
                          tile_assignment: np.ndarray, tensor_rank: int,
                          replicate_dims: Set[int]) -> Tuple[List, List]:
  group_assignment = list()
  replication_groups = list()
  if sharding_type is ShardingType.PARTIAL:
    # Shard across groups and replicate within subgroups; replicated dims
    # will be used to group replication devices.
    tile_shape = tile_assignment.shape
    # When creating the tile assignment, the mesh is permuted so that the first
    # few axes are used for tiling.
    tile_dims = range(tensor_rank - len(replicate_dims))
    group_list = [tile_assignment]
    for d in tile_dims:
      _group_list = list()
      for group_members in group_list:
        _group_list += np.split(group_members, tile_shape[d], d)
      group_list = _group_list
    replication_groups = [group.flatten().tolist() for group in group_list]

    mesh_axis = itertools.count()
    group_tile_shape = [
        1 if d in replicate_dims else tile_shape[next(mesh_axis)]
        for d in range(tensor_rank)
    ]
    group_assignment = np.arange(len(replication_groups)).reshape(
        tuple(group_tile_shape)).tolist()
  return group_assignment, replication_groups


def _translate_named_partition_spec(mesh: Mesh, partition_spec: Tuple):
  _partition_spec = list()
  for p in partition_spec:
    if type(p) is tuple:
      assert not any(type(x) is tuple
                     for x in p), 'Partition spec cannot contain nested tuples'
      _partition_spec.append(_translate_named_partition_spec(mesh, p))
    elif (p is None) or (type(p) is int):
      _partition_spec.append(p)
    elif type(p) is str:
      idx = mesh.get_axis_name_idx(p)
      if idx is None:
        raise ValueError(f"Axis name {p} is not defined in the given mesh")
      _partition_spec.append(idx)
    else:
      raise ValueError(
          f"Spec type {type(p)} is not supported in partition spec")
  return tuple(_partition_spec)


def _mark_manual_sharding(
    t: Union[torch.Tensor, XLAShardedTensor]) -> XLAShardedTensor:
  """
  This API is meant to be paired with the upcoming pause_spmd&resume_spmd APIs.
  Don't use it alone.
  """
  manual_sharding = torch_xla._XLAC.OpSharding([], [], [], ShardingType.MANUAL)
  torch_xla._XLAC._mark_manual_sharding(
      unwrap_sharded_tensor(t), manual_sharding)
  return wrap_as_sharded_tensor(t)


def enable_manual_sharding(t: Union[torch.Tensor, XLAShardedTensor],
                           partition_spec: Tuple[Union[Tuple, int, str, None]],
                           *,
                           mesh: Mesh = None) -> XLAShardedTensor:
  """
  This API enables manual sharding for the given tensor. Manual sharding disables SPMD sharding proporgation and auto
  partition for the given tensor and all subsequential tensors that produced by an op that uses the given tensor as
  input, and therefore allows the user to manually call collectives for the tensor and subsequential tensors. It
  requires the user to provide the partition spec to shard the tensor before enabling the manual sharding. To be noted,
  the leaf tensors need to pass to disable_manual_sharding before ending the graph.
  """
  mesh = get_global_mesh() if mesh is None else mesh
  t = mark_sharding(unwrap_sharded_tensor(t), mesh, partition_spec)
  t = torch_xla._XLAC._spmd_full_to_shard_shape(unwrap_sharded_tensor(t))
  return wrap_as_sharded_tensor(t)


def disable_manual_sharding(t: Union[torch.Tensor, XLAShardedTensor],
                            partition_spec: Tuple[Union[Tuple, int, str, None]],
                            full_shape: torch.Size,
                            *,
                            mesh: Mesh = None) -> XLAShardedTensor:
  """
  This API disables manual sharding for the given tensor. The partition_spec and full_shape are used to construct the
  output tensor as if the input tensor has not been manual sharded.
  """
  mesh = get_global_mesh() if mesh is None else mesh
  t = _mark_manual_sharding(unwrap_sharded_tensor(t))
  t = torch_xla._XLAC._spmd_shard_to_full_shape(
      unwrap_sharded_tensor(t), mesh.get_op_sharding(partition_spec),
      full_shape, t.dtype)
  return wrap_as_sharded_tensor(t)


def mark_sharding(
    t: Union[torch.Tensor, XLAShardedTensor], mesh: Mesh,
    partition_spec: Tuple[Union[Tuple, int, str, None],
                          ...]) -> XLAShardedTensor:
  """
    Annotates the tensor provided with XLA partition spec. Internally,
    it annotates the corresponding XLATensor as sharded for the XLA SpmdPartitioner pass.

    Args:
        t (Union[torch.Tensor, XLAShardedTensor]): input tensor to be annotated with partition_spec.

        mesh (Mesh): describes the logical XLA device topology and the underlying device IDs.

        partition_spec (Tuple[Tuple, int, str, None]): A tuple of device_mesh dimension index or
            `None`. Each index is an int, str if the mesh axis is named, or tuple of int or str.
            This specifies how each input rank is sharded (index to mesh_shape) or replicated (None).
            When a tuple is specified, the corresponding input tensor axis will be sharded along all
            logical axes in the tuple. Note that the order the mesh axes are specified in the tuple
            will impact the resulting sharding.

        dynamo_custom_op (bool): if set to True, it calls the dynamo custom op variant of mark_sharding
          to make itself recognizeable and traceable by dynamo.

    Example:

      >>> import torch_xla.runtime as xr
      >>> import torch_xla.distributed.spmd as xs
      >>> mesh_shape = (4, 2)
      >>> num_devices = xr.global_runtime_device_count()
      >>> device_ids = np.array(range(num_devices))
      >>> mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
      >>> input = torch.randn(8, 32).to(xm.xla_device())
      >>> xs.mark_sharding(input, mesh, (0, None)) # 4-way data parallel
      >>> linear = nn.Linear(32, 10).to(xm.xla_device())
      >>> xs.mark_sharding(linear.weight, mesh, (None, 1)) # 2-way model parallel
  """
  num_devices = xr.global_runtime_device_count()
  assert num_devices > 0, "This requires XLA supported device(s)."
  assert mesh.size() == num_devices, \
    f"{mesh.mesh_shape} is not mappable over {num_devices} devices."
  # We only allow fully specified `partition_spec` to be applicable, as opposed
  # to filling in the unspecified replicated dims. Fully specified `partiion_spec`
  # should be of the same rank as `t`. This is to support partial replication
  # where the group assignment may vary with different input ranks.
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) should be equal to the input rank ({len(t.shape)})."

  op_sharding = mesh.get_op_sharding(partition_spec)
  annotate_func = torch_xla._XLAC._xla_mark_sharding
  annotate_func(unwrap_sharded_tensor(t), op_sharding)
  return wrap_as_sharded_tensor(t)


def clear_sharding(t: Union[torch.Tensor, XLAShardedTensor]) -> torch.Tensor:
  """
  Clear sharding annotation from the input tensor and return a `cpu` casted tensor. This
  is a in place operation but will also return the same torch.Tensor back.

  Args:
    t (Union[torch.Tensor, XLAShardedTensor]): Tensor that we want to clear the sharding

  Return:
    t (torch.Tensor): tensor that without sharding.

  Example:

    >>> import torch_xla.distributed.spmd as xs
    >>> torch_xla.runtime.use_spmd()
    >>> t1 = torch.randn(8,8).to(torch_xla.device())
    >>> mesh = xs.get_1d_mesh()
    >>> xs.mark_sharding(t1, mesh, (0, None))
    >>> xs.clear_sharding(t1)
  """
  torch_xla._XLAC._xla_clear_sharding(unwrap_sharded_tensor(t))
  if isinstance(t, XLAShardedTensor):
    return t.global_tensor
  return t


def wrap_as_sharded_tensor(
    t: Union[torch.Tensor, XLAShardedTensor]) -> XLAShardedTensor:
  if not isinstance(t, XLAShardedTensor):
    return XLAShardedTensor(t)
  return t


def unwrap_sharded_tensor(
    t: Union[torch.Tensor, XLAShardedTensor]) -> torch.Tensor:
  if isinstance(t, XLAShardedTensor):
    return t.global_tensor
  return t


def wrap_if_sharded(x: Any) -> Any:
  """
  If the input is a sharded tensor, return an XLAShardedTensor wrapping it.
  Otherwise, returns the input.
  """
  if (isinstance(x, torch.Tensor) and not isinstance(x, XLAShardedTensor) and
      x.device.type == 'xla' and
      torch_xla._XLAC._get_xla_sharding_type(x) is not None):
    return XLAShardedTensor(x)
  return x


@dataclass
class ShardingSpec:
  mesh: Mesh
  partition_spec: Tuple[Union[int, None]]
  minibatch: Optional[bool] = False

  # Derived fields
  _tile_assignment: List[int] = field(init=False)
  _group_assignment: List[int] = field(init=False)
  _replication_groups: List[int] = field(init=False)
  _sharding_type: ShardingType = field(init=False)

  def __post_init__(self):
    mesh = self.mesh
    partition_spec = _translate_named_partition_spec(mesh, self.partition_spec)
    tile_assignment = _get_tile_assignment(mesh, partition_spec)
    self._tile_assignment = tile_assignment.tolist()
    self._sharding_type = _get_sharding_type(partition_spec,
                                             xr.global_runtime_device_count())
    replicate_dims = {i for i, d in enumerate(partition_spec) if d is None}
    self._group_assignment, self._replication_groups = _get_group_assignment(
        self._sharding_type, tile_assignment, len(partition_spec),
        replicate_dims)

  def xla_spec(self, t: torch.Tensor) -> Union['XlaShardingSpec', None]:
    """
    Create an XlaShardingSpec for the given tensor. If the tensor is
    incompatible with the ShardingSpec, returns None.
    """
    if not self.can_apply(t):
      return None
    return torch_xla._XLAC.XlaShardingSpec(t, self._tile_assignment,
                                           self._group_assignment,
                                           self._replication_groups,
                                           int(self._sharding_type),
                                           self.minibatch)

  def can_apply(self, t: torch.Tensor) -> bool:
    """
    Test whether the ShardingSpec is compatible with the given torch.Tensor.
    """
    return len(t.shape) == len(self.partition_spec)

  def apply(self, t: torch.Tensor):
    # TODO(yeounoh) use virtual device interface when available.
    assert (t.device == xm.xla_device())
    mark_sharding(t, self.mesh, self.partition_spec)


class XLAPatchedLinear(torch.autograd.Function):
  """
  A patched version of `torch.nn.functional.linear` that uses einsum instead
  of torch.matmul which will flatten the tensors to 2D and collide the sharded
  dimensions. The torch.matmul default behavior makes it very hard for XLA compiler
  to propagate the sharding annotation.

  Autocast decorators @custom_fwd and @custom_bwd used as per autocast docs [1] to bring this class/layer within 
  autocast context, when autocast is enabled.
  torch.get_autocast_dtype() fetches datatype for ops run in autocast [2], with the specified device (here, 'xla').
  
  References: 
  [1] https://pytorch.org/docs/stable/notes/amp_examples.html#functions-with-multiple-inputs-or-autocastable-ops 
  [2] https://github.com/pytorch/pytorch/blob/2cc01cc6d3ad2aff47e8460667ba654b2e4c9f21/torch/amp/autocast_mode.py#L500

  TODO (alanwaketan): Let's patch it on the dispatcher level.
  """

  @staticmethod
  @custom_fwd(device_type='xla', cast_inputs=torch.get_autocast_dtype('xla'))
  def forward(ctx, input, weight, bias=None):
    # bias is an optional argument
    ctx.save_for_backward(input, weight, bias)
    with torch.no_grad():
      product = torch.einsum('...n,mn->...m', input, weight)
      if bias is None:
        return product
      return product + bias

  @staticmethod
  @custom_bwd(device_type='xla')
  def backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    if ctx.needs_input_grad[0]:
      grad_input = torch.einsum('...m,mn->...n', grad_output, weight)
    if ctx.needs_input_grad[1]:
      grad_weight = torch.einsum('...m,...n->mn', grad_output, input)
    if bias is not None and ctx.needs_input_grad[2]:
      grad_bias = torch.einsum('...m->m', grad_output)

    return grad_input, grad_weight, grad_bias


def xla_patched_nn_linear_forward(m, input):
  return XLAPatchedLinear.apply(input, m.weight, m.bias)


def apply_backward_optimization_barrier(m: torch.nn.Module):
  """
  Register a full backward hook that apply an optimization barrier to the given module.
  This will prevent the XLA compiler from fusing the module's backward pass with others.
  It's useful to prevent gigantic buffers being allocated to synchronize the gradients.
  """

  def optimization_barrier(module, grad_input, grad_output):
    from torch_xla.utils.checkpoint import CheckpointFunction
    gradients = []
    for param in module.parameters():
      if param.grad != None:
        gradients.append(param.grad)
    xm.optimization_barrier_(
        CheckpointFunction._extract_tensors_from_list(gradients +
                                                      list(grad_input)))

  m.register_full_backward_hook(optimization_barrier)
