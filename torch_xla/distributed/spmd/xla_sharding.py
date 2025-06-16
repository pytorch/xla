import collections
from collections.abc import Generator, MutableMapping
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import torch
from torch import Tensor
from torch.library import custom_op
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla._internal.utils as _utils
from torch_xla.distributed.spmd import XLAShardedTensor, XLAShard
import torch_xla.runtime as xr
import torch_xla.debug.profiler as xp
from torch_xla._internal.jax_workarounds import requires_jax, maybe_get_torchax

import numpy as np
import functools
import itertools
from typing import TypeVar, Union, Any, Optional
from collections.abc import Sequence
from enum import IntEnum

from torch.amp import custom_fwd, custom_bwd
from torch.utils._pytree import tree_flatten, tree_unflatten

PartitionSpec = tuple[Union[tuple[Union[int, str], ...], int, str, None], ...]
"""PartitionSpec describes the sharding of a tensor.

Specifically, it is a tuple of one or more device mesh axes that describes how to
shard the input tensor. For example, the first dimension of the tensor is sharded
across the axis/axes described in the first element of this tuple and so on.
"""


class Mesh:
  """Describe the logical XLA device topology mesh and the underlying resources.

  Args:
    device_ids: A flattened list of devices (IDs).
        The list is reshaped to an array of shape `mesh_shape`, filling the elements using
        row-major order. Each ID indexes into the list of devices returned by
        `xr.global_runtime_device_attributes()`.

    mesh_shape: A int tuple describing the shape of the device mesh. Each element
        describes the number of devices in the corresponding axis.

    axis_names: A sequence of mesh axis names. Its length should match the length of `mesh_shape`.

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
  mesh_shape: tuple[int, ...]
  axis_names: Optional[tuple[str, ...]]

  def __init__(self,
               device_ids: Union[np.ndarray, list[int]],
               mesh_shape: tuple[int, ...],
               axis_names: Optional[tuple[str, ...]] = None):
    if not isinstance(device_ids, np.ndarray):
      device_ids = np.array(device_ids)

    # At the moment, XLA requires that the Mesh uses the global number of
    # devices.
    num_devices = xr.global_runtime_device_count()
    assert num_devices > 0, "This requires XLA supported device(s)."
    assert num_devices == len(
        device_ids
    ), f"Number of device IDs ({len(device_ids)}) must match the global number of devices ({num_devices})"

    if axis_names is not None:
      assert len(mesh_shape) == len(axis_names), \
          f"Number of axis names ({len(axis_names)}) must match mesh dimensions ({len(mesh_shape)})"
      assert len(set(axis_names)) == len(axis_names), \
          f"Axis names must be unique, got: {axis_names}"

    expected_devices = np.prod(mesh_shape)
    assert len(device_ids) == expected_devices, \
        f"Number of device IDs ({len(device_ids)}) must match mesh size ({expected_devices})"
    assert len(device_ids) == len(np.unique(device_ids)), \
        f"Device IDs must be unique, got: {device_ids}"

    self.device_ids = device_ids
    self.mesh_shape = mesh_shape
    self.axis_names = axis_names
    assert all(d < self.size() for d in device_ids), \
        f"Device IDs must be less than mesh size ({self.size()}), got: {device_ids}"

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
  def _get_op_sharding_args(self, partition_spec: PartitionSpec):
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
  def get_op_sharding(
      self, partition_spec: PartitionSpec) -> torch_xla._XLAC.OpSharding:
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

  def __str__(self):
    """Convert Mesh to string representation."""
    return (f"{{'device_ids': {self.device_ids.tolist()}, "
            f"'mesh_shape': {self.mesh_shape}, "
            f"'axis_names': {self.axis_names}}}")

  @classmethod
  def from_str(cls, mesh_str: str) -> Optional["Mesh"]:
    """Create Mesh from string representation."""
    import ast
    import numpy as np
    try:
      dict_str = mesh_str.replace('Mesh', '')
      mesh_dict = ast.literal_eval(dict_str)
      return cls(
          device_ids=np.array(mesh_dict['device_ids']),
          mesh_shape=mesh_dict['mesh_shape'],
          axis_names=mesh_dict['axis_names'])
    except (ValueError, SyntaxError, KeyError, TypeError):
      return None

  @requires_jax
  def get_jax_mesh(self):
    # Construct a JAX mesh object with the same device ids shape and ordering
    # from torch_xla device mesh.
    import jax
    import numpy as np
    from jax._src import mesh as mesh_lib

    axis_names = self.axis_names or tuple(
        str(i) for i in range(len(self.mesh_shape)))

    # Create a mapping from device ID to device object
    all_devices = jax.devices()
    device_id_to_device = {device.id: device for device in all_devices}
    device_ids_array = self.device_ids.reshape(*self.mesh_shape)
    device_array = np.empty(device_ids_array.shape, dtype=object)
    device_array = np.vectorize(device_id_to_device.get)(device_ids_array)
    if np.any(device_array == None):
      raise ValueError(
          f"torch_xla device ID {device_ids_array[device_array == None]} not found in available JAX devices"
      )
    return mesh_lib.Mesh(device_array, axis_names=axis_names)


_GLOBAL_MESH: Optional[Mesh] = None


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
  ici_mesh_shape: tuple[int, ...]
  dcn_mesh_shape: tuple[int, ...]

  def __init__(self,
               *,
               ici_mesh_shape: tuple[int, ...],
               dcn_mesh_shape: Optional[tuple[int, ...]] = None,
               axis_names: Optional[tuple[str, ...]] = None):
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
  def _get_physical_tpu_mesh(self, devices: np.ndarray) -> np.ndarray:
    r"""Rearrange TPU devices in a slice into a physical mesh.

      Args:
        devices: A list of device logical ordinals in a TPU slice.

      Returns:
        A np.ndarray of device logical ordinals with shape [global_x, global_y, global_z]. On
          v2 and v3, global_z is instead cores_per_chip (i.e., 2).
    """
    assert xm.xla_device_hw(torch_xla.device()) == 'TPU'
    # coords is a 3-dims tuple representing the device in physical mesh
    device_coords = [self.device_attributes[d]['coords'] for d in devices]
    dims = tuple(d + 1 for d in max(device_coords))
    out = np.empty(dims, dtype=int)
    for coords, d in zip(device_coords, devices):
      out[coords[0], coords[1], coords[2]] = d
    return out

  def _create_device_mesh(self,
                          mesh_shape: Sequence[int],
                          devices: Optional[np.ndarray] = None) -> np.ndarray:
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
    device_mesh, assignment = _create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape, allow_split_physical_axes=True)
    return device_mesh

  # This is imported from JAX: https://github.com/google/jax/blob/main/jax/experimental/mesh_utils.py#L288.
  def _create_hybrid_device_mesh(self, ici_mesh_shape: Sequence[int],
                                 dcn_mesh_shape: Sequence[int]) -> np.ndarray:
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
    # creates a separate internal mesh for each slice.
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
  # ShardingType enum ID maps to OpSharding.Type (https://shorturl.at/pvAJX)
  REPLICATED = 0
  MAXIMAL = 1
  TUPLE = 2
  TILED = 3
  MANUAL = 4
  PARTIAL = 5
  UNKNOWN = 6  # implicit replication. TODO(yeounoh) wait for auto-sharding support


def _get_sharding_type(partition_spec: PartitionSpec,
                       num_devices: int) -> ShardingType:
  sharding_type = ShardingType.TILED
  if num_devices == 1:
    sharding_type = ShardingType.MAXIMAL
  elif all(d is None for d in partition_spec):
    sharding_type = ShardingType.REPLICATED
  elif any(d is None for d in partition_spec):
    sharding_type = ShardingType.PARTIAL
  return sharding_type


def _get_tile_assignment(mesh: Mesh,
                         partition_spec: PartitionSpec) -> np.ndarray:
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
                          replicate_dims: set[int]) -> tuple[list, list]:
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


def _translate_named_partition_spec(mesh: Mesh, partition_spec: PartitionSpec):
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
                           partition_spec: PartitionSpec,
                           *,
                           mesh: Optional[Mesh] = None) -> XLAShardedTensor:
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
                            partition_spec: PartitionSpec,
                            full_shape: torch.Size,
                            *,
                            mesh: Optional[Mesh] = None) -> XLAShardedTensor:
  """
  This API disables manual sharding for the given tensor. The partition_spec and full_shape are used to construct the
  output tensor as if the input tensor has not been manually sharded.
  """
  mesh = get_global_mesh() if mesh is None else mesh
  t = _mark_manual_sharding(unwrap_sharded_tensor(t))
  t = torch_xla._XLAC._spmd_shard_to_full_shape(
      unwrap_sharded_tensor(t), mesh.get_op_sharding(partition_spec),
      full_shape, t.dtype)
  return wrap_as_sharded_tensor(t)


def annotate_custom_sharding(t: Union[torch.Tensor,
                                      XLAShardedTensor], mesh: Mesh,
                             partition_spec: PartitionSpec) -> XLAShardedTensor:
  """
  Annotates an existing tensor with a custom sharding IR node without modifying its data layout.

  Unlike `mark_sharding`, this function only adds a custom sharding annotation to the XLA IR
  without explicitly setting a sharding spec tied to the DeviceData node or transferring any
  sharded data to the device. This allows providing explicit XLA sharding annotations of tensors
  that have already been sharded with `mark_sharding`.

  Args:
      t: The input tensor to be annotated with custom sharding.
      mesh: The device mesh that specifies the logical device topology.
      partition_spec: The partitioning specification for each dimension of the input tensor.

  Returns:
      XLAShardedTensor: The input tensor wrapped as a sharded tensor with the custom sharding annotation.

  Example:
      >>> # First shard the tensor with mark_sharding
      >>> sharded_tensor = xs.mark_sharding(tensor, mesh1, (0, 1, 2, 3))
      >>> # Later, annotate with a different sharding for the XLA SPMD partitioner
      >>> custom_sharded = xs.annotate_custom_sharding(sharded_tensor, mesh2, (0, 1, 2, 3))
  """
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) should be equal to the input rank ({len(t.shape)})."

  op_sharding = mesh.get_op_sharding(partition_spec)
  annotate_func = torch_xla._XLAC._xla_annotate_custom_sharding
  annotate_func(unwrap_sharded_tensor(t), op_sharding)
  return wrap_as_sharded_tensor(t)


def mark_sharding(t: Union[torch.Tensor, XLAShardedTensor], mesh: Mesh,
                  partition_spec: PartitionSpec) -> XLAShardedTensor:
  """
    Annotates the tensor provided with XLA partition spec. Internally,
    it annotates the corresponding XLATensor as sharded for the XLA SpmdPartitioner pass.

    Args:
        t (Union[torch.Tensor, XLAShardedTensor]): input tensor to be annotated with partition_spec.

        mesh (Mesh): describes the logical XLA device topology and the underlying device IDs.

        partition_spec (PartitionSpec): A tuple of one or more device mesh axes that describes how
            to shard the input tensor. Each element can be:

            - an int: refer to a mesh axis by index
            - str: refer to a mesh axis by name
            - a tuple of the above: refer to multiple mesh axes
            - None: the corresponding tensor dimension will be replicated over all devices

            This specifies how each input rank is sharded (index to mesh_shape) or replicated (None).
            When a tuple is specified, the corresponding input tensor axis will be sharded along all
            mesh axes in the tuple. Note that the order the mesh axes are specified in the tuple
            will impact the resulting sharding.

    Example:

      >>> import torch_xla.runtime as xr
      >>> import torch_xla.distributed.spmd as xs
      >>> mesh_shape = (4, 2)
      >>> num_devices = xr.global_runtime_device_count()
      >>> device_ids = np.array(range(num_devices))
      >>> mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
      >>> input = torch.randn(8, 32).to('xla')
      >>> xs.mark_sharding(input, mesh, (0, None)) # 4-way data parallel
      >>> linear = nn.Linear(32, 10).to('xla')
      >>> xs.mark_sharding(linear.weight, mesh, (None, 1)) # 2-way model parallel
  """
  # We only allow fully specified `partition_spec` to be applicable, as opposed
  # to filling in the unspecified replicated dims. Fully specified `partition_spec`
  # should be of the same rank as `t`. This is to support partial replication
  # where the group assignment may vary with different input ranks.
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) should be equal to the input rank ({len(t.shape)})."

  tx = maybe_get_torchax()
  if tx is not None and isinstance(t, tx.tensor.Tensor):
    from jax.sharding import PartitionSpec as P, NamedSharding
    jmesh = mesh.get_jax_mesh()
    t.shard_(NamedSharding(jmesh, P(*partition_spec)))
    return t

  op_sharding = mesh.get_op_sharding(partition_spec)
  annotate_func = torch_xla._XLAC._xla_mark_sharding
  annotate_func(unwrap_sharded_tensor(t), op_sharding)
  return wrap_as_sharded_tensor(t)


def mark_sharding_with_gradients(
    t: Union[torch.Tensor, XLAShardedTensor], mesh: Mesh,
    partition_spec: tuple[Union[tuple, int, str, None], ...]) -> torch.Tensor:
  """
    A function to add sharding annotations on intermediate tensors (not in-place) and the gradient
    of the intermediate tensors during backward pass.

    Args:
        t (Union[torch.Tensor, XLAShardedTensor]): input tensor to be annotated with partition_spec.

        mesh (Mesh): describes the logical XLA device topology and the underlying device IDs.

        partition_spec (PartitionSpec): A tuple of one or more device mesh axes that describes how
            to shard the input tensor. Each element can be:

    Usage:

    >>> new_tensor = MarkShardingFunction.apply(tensor, mesh, ('axis_1', 'axis_2'))
    sharding annotations are added to `new_tensor` and `tensor.grad`.

    This is required to guide GSPMD sharding propagation better during the
    backward pass as during complicated workloads the compiler can introduce extra
    collectives that can hurt performance.

    Compared to `mark_sharding`, this version will not in-place shard input tensors.
    Instead it takes in an unsharded tensor and returns a new tensor that is sharded.
    After GSPMD sharding propagation in the compiler, both tensors will become sharded.

    This version can also be used in AOTAutograd.
    """
  # We only allow fully specified `partition_spec` to be applicable, as opposed
  # to filling in the unspecified replicated dims. Fully specified `partition_spec`
  # should be of the same rank as `t`. This is to support partial replication
  # where the group assignment may vary with different input ranks.
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) should be equal to the input rank ({len(t.shape)})."

  r = MarkShardingFunction.apply(t, mesh, partition_spec)
  assert isinstance(r, torch.Tensor)
  return r


PyTreeA = TypeVar('PyTreeA')
PyTreeB = TypeVar('PyTreeB')


@requires_jax
def shard_as(a: PyTreeA, b: PyTreeB) -> tuple[PyTreeA, PyTreeB]:
  """Ensure that `a` and `b` are sharded the same way without specifying
  a particular sharding constraint.

  shard_as takes two PyTrees of matching structure and returns
  two PyTrees of that same structure. As long as you use at least one
  of the outputs, then corresponding tensors in all four PyTrees
  (a, b, out[0], out[1]) will be sharded the same way.
  """

  a_flat, a_spec = tree_flatten(a)
  b_flat, b_spec = tree_flatten(b)
  assert a_spec == b_spec, f"a and b must have the same structure. got {a_spec} and {b_spec}"
  a_sharded_flat = []
  b_sharded_flat = []
  from jax.experimental.shard_alike import shard_alike
  for x, y in zip(a_flat, b_flat):
    if x is None or y is None:
      # If there are None leaves, then it should be None in both PyTrees.
      assert x is None and y is None
    else:
      x, y = xb.call_jax(shard_alike, (x, y))
    a_sharded_flat.append(x)
    b_sharded_flat.append(y)
  return tree_unflatten(a_sharded_flat,
                        a_spec), tree_unflatten(b_sharded_flat, b_spec)


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
    >>> t1 = torch.randn(8,8).to('xla')
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
  partition_spec: PartitionSpec
  minibatch: Optional[bool] = False

  # Derived fields
  _tile_assignment: list[int] = field(init=False)
  _group_assignment: list[int] = field(init=False)
  _replication_groups: list[int] = field(init=False)
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
    assert (t.device == torch_xla.device())
    mark_sharding(t, self.mesh, self.partition_spec)


### Linear layer implementation backed by einsum.


# A custom forward op that uses einsum internally
@custom_op(
    "xla::einsum_linear_forward",
    schema="(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
    mutates_args=())
def _einsum_linear_forward(input: Tensor, weight: Tensor,
                           bias: Optional[Tensor]):
  with xp.Trace('einsum_linear_forward'):
    product = torch.einsum('...n,mn->...m', input, weight)
    if bias is not None:
      return product + bias
    return product


@_einsum_linear_forward.register_fake
def _einsum_linear_forward_fake(input: Tensor, weight: Tensor,
                                bias: Optional[Tensor]):
  product = torch.einsum('...n,mn->...m', input, weight)
  if bias is not None:
    return product + bias
  return product


def _einsum_linear_backward_operation(grad_output: Tensor, input: Tensor,
                                      weight: Tensor, bias: Optional[Tensor],
                                      needs_input_grad_input: bool,
                                      needs_input_grad_weight: bool,
                                      needs_input_grad_bias: bool):
  grad_input = grad_weight = grad_bias = None

  if needs_input_grad_input:
    grad_input = torch.einsum('...m,mn->...n', grad_output, weight).clone()
  else:
    grad_input = None

  if needs_input_grad_weight:
    grad_weight = torch.einsum('...m,...n->mn', grad_output, input).clone()
  else:
    grad_weight = None

  if bias is not None and needs_input_grad_bias:
    grad_bias = torch.einsum('...m->m', grad_output).clone()
  else:
    grad_bias = None

  return grad_input, grad_weight, grad_bias


@custom_op(
    "xla::einsum_linear_backward",
    schema="(Tensor grad_output, Tensor input, Tensor weight, Tensor? bias, bool needs_input_grad_input, bool needs_input_grad_weight, bool needs_input_grad_bias) -> (Tensor, Tensor, Tensor)",
    mutates_args=())
def _einsum_linear_backward(grad_output: Tensor, input: Tensor, weight: Tensor,
                            bias: Optional[Tensor],
                            needs_input_grad_input: bool,
                            needs_input_grad_weight: bool,
                            needs_input_grad_bias: bool):
  with xp.Trace('einsum_linear_backward'):
    return _einsum_linear_backward_operation(grad_output, input, weight, bias,
                                             needs_input_grad_input,
                                             needs_input_grad_weight,
                                             needs_input_grad_bias)


@_einsum_linear_backward.register_fake
def _einsum_linear_backward_fake(grad_output: Tensor, input: Tensor,
                                 weight: Tensor, bias: Optional[Tensor],
                                 needs_input_grad_input: bool,
                                 needs_input_grad_weight: bool,
                                 needs_input_grad_bias: bool):

  return _einsum_linear_backward_operation(grad_output, input, weight, bias,
                                           needs_input_grad_input,
                                           needs_input_grad_weight,
                                           needs_input_grad_bias)


# Now define the XLAPatchedLinear function that uses the custom ops
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
  def forward(ctx,
              input: Tensor,
              weight: Tensor,
              bias: Optional[Tensor] = None):
    ctx.save_for_backward(input, weight, bias)
    # Call our custom forward op. By wrapping the einsum in custom ops,
    # AOTAutograd won't decompose the einsum.
    return torch.ops.xla.einsum_linear_forward(input, weight, bias)

  @staticmethod
  @custom_bwd(device_type='xla')
  def backward(ctx, grad_output: Tensor):
    input, weight, bias = ctx.saved_tensors
    needs_input_grad_input = ctx.needs_input_grad[0]
    needs_input_grad_weight = ctx.needs_input_grad[1]
    needs_input_grad_bias = False
    if bias is not None:
      needs_input_grad_bias = ctx.needs_input_grad[2]

    # Call our custom backward op with the boolean flags
    grad_input, grad_weight, grad_bias = torch.ops.xla.einsum_linear_backward(
        grad_output, input, weight, bias, needs_input_grad_input,
        needs_input_grad_weight, needs_input_grad_bias)
    return grad_input, grad_weight, grad_bias, None


def xla_patched_nn_linear_forward(m, input):
  return XLAPatchedLinear.apply(input, m.weight, m.bias)


class EinsumLinear(torch.nn.Linear):
  """
  A `torch.nn.Linear` subclass implemented with `einsum`.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, input):
    t = xla_patched_nn_linear_forward(self, input)
    assert isinstance(t, torch.Tensor)
    return t


def apply_xla_patch_to_nn_linear(module: torch.nn.Module):
  """
  Recursively replace `nn.Linear` layers with `EinsumLinear` in the module.

  Without this patch, an `nn.Linear` module in PyTorch/XLA will lower to reshapes
  and transposes instead of einsum, thus compromising sharding propagation.
  """
  for name, child in module.named_children():
    if isinstance(child,
                  torch.nn.Linear) and not isinstance(child, EinsumLinear):
      with torch.device('meta'):
        einsum_linear = EinsumLinear(
            child.in_features, child.out_features, bias=child.bias is not None)
      einsum_linear.load_state_dict(
          child.state_dict(), strict=True, assign=True)
      setattr(module, name, einsum_linear)
    else:
      apply_xla_patch_to_nn_linear(child)

  return module


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


###############################################################################
#
# The following is copied from JAX: https://github.com/jax-ml/jax/blob/main/jax/_src/mesh_utils.py
#
###############################################################################


def _create_device_mesh_for_nd_torus(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
    *,
    allow_split_physical_axes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Assigns logical parallelism axes to physical axes of an N-D torus network.

  Given logical parallelism axes with sizes in `mesh_shape` and devices in an
  N-dimensional torus network represented by `physical_mesh`, maps each logical
  axis to one or more physical axes. Prefer to map more-performance-sensitive
  logical axes to larger numbers of physical axes to maximize the bandwidth
  available to them. Also prefer to assign logical axes to multiple physical
  axes of the same size (e.g., a 2D square) rather than multiple physical axes
  of different sizes when possible.

  If allow_split_physical_axes = False (default), this routine will error out
  instead of splitting a physical axis over more than one logical axis (which
  would reduce total usable bandwidth).

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
    allow_split_physical_axes: If True, we would split physical axes if
      necessary to fit the desired mesh shape.

  Returns:
    An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
      each logical parallelism axis mapped to one or more physical mesh axes.
    The axis assignment matrix, which is a 2-d array mapping from
      (physical_axis, logical_axis) to the size assigned, with the invariant
      np.prod(assignment, axis=1) = physical_mesh_shape, and
      np.prod(assignment, axis=0) = mesh_shape.
  """
  # Remaining physical axes to be assigned to logical axes.
  assignable_physical_mesh = list(physical_mesh.shape)
  # Map each logical axis to a subset of physical axes.
  assignment: list[tuple[int, ...]] = [() for _ in mesh_shape]

  # Assign logical axes from highest network intensity to lowest.
  # `mesh_shape` is assumed to ordered by lowest network intensity first, so
  # reverse it first.
  for logical_axis_index, logical_axis_size in reversed(
      list(enumerate(mesh_shape))):
    # Preferentially map to more physical axes first for higher bandwidth.
    for num_axes in range(3, 0, -1):
      # Try assign to any subset of size num_axes. Generate all candidates.
      indices_and_axes = itertools.combinations(
          enumerate(assignable_physical_mesh), num_axes)
      for elem in indices_and_axes:
        c_indices, c_axes = zip(*elem)
        # TODO(zhangqiaorjc): Due to limitations in XLA, 2D collectives only
        # implemented for square 2D plane. Mapping a physical axis to two
        # logical axes might be slower for non-square 2D plane, e.g., map 32 to
        # 4x8 or a single axis. If XLA 2D collectives support non-square plane
        # soon, we can continue to preferentially map to 2D plane in general,
        # otherwise, we should treat non-square 2D plane and 1D submesh equally.
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
        if not allow_split_physical_axes:
          # Although this is now implemented, there are downstream tasks
          # counting on this being a NotImplementedError.
          raise NotImplementedError(
              'Failed to find assignment for logical_axis_index'
              f' {logical_axis_index} of size {logical_axis_size} with'
              f' remaining assignable mesh {assignable_physical_mesh}. The size'
              ' of each axis in your logical mesh must be equal to the product'
              ' of some subset of the physical mesh axis sizes. E.g. logical'
              ' mesh (4, 16) is compatible with physical mesh 4x4x4 since 4=4'
              ' and 16=4x4. If you want to split physical axes, set '
              ' allow_split_physical_axes to True.')
        else:
          # We will try finding an assignment, even if that means splitting the
          # physical axes, which requires a more sophisticated implementation.
          return _create_device_mesh_for_nd_torus_splitting_axes(
              physical_mesh, mesh_shape)

  # Flatten the assignment, e.g., [(), (2,), (0, 1)] -> (2, 0, 1).
  transpose: list[int] = []
  assignment_array = np.ones(
      [len(physical_mesh.shape), len(mesh_shape)], dtype=np.int64)
  for i, x in enumerate(assignment):
    for y in x:
      physical_mesh_axis = int(y)
      assignment_array[physical_mesh_axis,
                       i] = physical_mesh.shape[physical_mesh_axis]
      transpose.append(physical_mesh_axis)
  return (
      physical_mesh.transpose(transpose).reshape(mesh_shape),
      assignment_array,
  )


def _prefer_first_logical_axis_assignment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> bool:
  """Returns True if the first axis assignment is preferred over the second.

  For now, this is implemented with some very simple heuristics. However,
  it is possible to introduce e.g., a value function here based on a more
  precise model of the underlying hardware.

  TODO(rosun): Use a proxy of network capacity to select the partitions.

  Args:
    x: Logical axis assignment as [len(physical_mesh_shape)] array.
    y: Logical axis assignment as [len(physical_mesh_shape)] array.
    physical_mesh_shape: Physical mesh shape.
    assignment: Assignment matrix.

  Returns:
    True if x is preferred over y.
  """
  # Prefer occupying complete physical axes. I don't have a good reason for
  # this, except that it is compatible with the existing behavior.
  #
  # E.g., on 4 x 4 x 8, [4, 4, -] will be preferred over [4, -, 4], and then
  # over [2, 2, 4].
  x_whole_axis_size = np.prod(
      [s for i, s in enumerate(x) if s == physical_mesh_shape[i]])
  y_whole_axis_size = np.prod(
      [s for i, s in enumerate(y) if s == physical_mesh_shape[i]])

  if x_whole_axis_size != y_whole_axis_size:
    return x_whole_axis_size > y_whole_axis_size

  # Prefer occupying more whole physical axes for better bandwidth.
  #
  # This is consistent with existing logic, i.e., 2 x 2 is preferred over 4.
  x_num_whole_axes = len(
      [1 for i, s in enumerate(x) if s == physical_mesh_shape[i] and s > 1])
  y_num_whole_axes = len(
      [1 for i, s in enumerate(y) if s == physical_mesh_shape[i] and s > 1])

  if x_num_whole_axes != y_num_whole_axes:
    return x_num_whole_axes > y_num_whole_axes

  # Prefer taking physical axes that are not taken by logical axes of higher
  # network intensity. E.g., for a 4 x 4 x 4, suppose that the previous
  # assignments are 1 x 2 x 4, and we want to place a new logical axis of size
  # 2, we will go for [2, 1, 1] instead of [1, 2, 1], as the latter choice will
  # tap into bandwidth already taken by the higher intensity axis.
  assigned_physical_mesh_shape = np.prod(assignment, axis=-1)

  x_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(x) if assigned_physical_mesh_shape[i] > 1])
  y_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(y) if assigned_physical_mesh_shape[i] > 1])

  if x_non_overlapping_axis_size != y_non_overlapping_axis_size:
    return x_non_overlapping_axis_size > y_non_overlapping_axis_size

  # Otherwise sort by reverse lexical graphical order, to be consistent with
  # existing behavior.
  return tuple(x) > tuple(y)


def _create_device_mesh_for_nd_torus_splitting_axes(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
  """Assigns logical parallelism axes to physical axes of an N-D torus network.

  This implementation allows creating meshes that requires splitting physical
  axes, and thus one could produce logical mesh of any shape, as long as the
  number of devices matches, e.g.,

  - Creating 2x2x4 from 4x4;

  - Creating 2x2x16 from 8x8;

  Args:
    physical_mesh: a np.ndarray of devices in the shape of the N-D torus
      physical topology.
    mesh_shape: shape of the logical mesh (size of the various logical
      parallelism axes), with axes ordered by increasing network intensity.

  Returns:
    An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
      each logical parallelism axis mapped to one or more physical mesh axes.
    The axis assignment matrix, which is a 2-d array mapping from
      (physical_axis, logical_axis) to the size assigned, with the invariant
      np.prod(assignment, axis=1) = physical_mesh_shape, and
      np.prod(assignment, axis=0) = mesh_shape.
  """
  if np.prod(physical_mesh.shape) != np.prod(mesh_shape):
    raise ValueError(
        'The number of devices in physical mesh'
        f' {physical_mesh.shape} does not match the number of devices'
        f' in logical mesh {mesh_shape}.')

  physical_mesh_shape = physical_mesh.shape
  logical_mesh_shape = tuple(mesh_shape)

  # (Partial) assignment map as an 2-d array [p_axis, l_axis] -> size.
  assignment = np.ones([len(physical_mesh_shape),
                        len(logical_mesh_shape)],
                       dtype=np.int64)

  # Process logical axes from highest network intensity to lowest.
  # `mesh_shape` is assumed to ordered by lowest network intensity first, so
  # reverse it.
  for logical_axis, logical_axis_size in reversed(
      list(enumerate(logical_mesh_shape))):
    # Go over all the possible assignment for the logical axis, including the
    # one that splits multiple physical axes.
    best_logical_axis_assignment = None
    for logical_axis_assignment in _enumerate_feasible_logical_axis_assignments(
        physical_mesh_shape, assignment, logical_axis_size):
      # TODO(rosun): Instead of using heuristics, replace this with a proper
      # scoring function reflecting the underlying hardware properties.
      if (best_logical_axis_assignment is None or
          _prefer_first_logical_axis_assignment(
              logical_axis_assignment,
              best_logical_axis_assignment,
              physical_mesh_shape=physical_mesh_shape,
              assignment=assignment,
          )):
        best_logical_axis_assignment = logical_axis_assignment
    assignment[:,
               logical_axis] = best_logical_axis_assignment  # type: ignore  # numpy 2.2

  # Read out the assignment.
  logical_mesh = _generate_logical_mesh(physical_mesh, logical_mesh_shape,
                                        assignment)

  return logical_mesh, assignment


def _get_prime_factors(x: int) -> list[int]:
  """Returns a sorted list of prime factors for the given number."""
  assert x > 0
  factors = []
  for p in range(2, math.isqrt(x) + 2):
    while x % p == 0:
      factors.append(p)
      x //= p
    if x == 1:
      return factors
  else:
    return [x]  # x is a prime number.


def _enumerate_feasible_logical_axis_assignments(
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
    logical_axis_size: int,
) -> Generator[np.ndarray, None, None]:
  """Yields feasible assignments for a single logical axis.

  For a physical mesh of shape [x_1, ..., x_n], and the product of all previous
  assignments on each physical axes [y_1, ..., y_n], this function yields all
  possible assignments for the axis as 1-d arrays [z_1, ..., z_n], so that:

  - prod(z_1, ..., z_n) = logical_axis_size

  - x_i % (z_i * y_i) = 0

  Args:
    physical_mesh_shape: Physical mesh shape.
    assignment: Existing assignment matrix.
    logical_axis_size: Size of the logical axis to assign.

  Yields:
    All valid assignments for the logical axis. Each assignment is represented
    as an integer array of length len(physical_mesh_shape).
  """
  logical_axis_factors: MutableMapping[int, int] = collections.defaultdict(int)
  for factor in _get_prime_factors(logical_axis_size):
    logical_axis_factors[factor] += 1

  available_physical_mesh_shape = np.array(physical_mesh_shape) // np.prod(
      assignment, axis=-1)

  # To enable efficient enumerations, we first index physical axes by their
  # prime factors. Since we know the prime factorization of the logical axis
  # size, we could simply enumerate by picking the correct count for each
  # prime factor.
  physical_axes_by_factor: MutableMapping[int, list[int]] = (
      collections.defaultdict(list))
  for physical_axis, physical_axis_size in enumerate(
      available_physical_mesh_shape):
    for factor in _get_prime_factors(physical_axis_size):
      if factor not in logical_axis_factors:
        continue
      physical_axes_by_factor[factor].append(physical_axis)

  factors = []
  assignments_by_factor = []
  for factor, multiplicity in logical_axis_factors.items():
    factors.append(factor)
    assignments_by_factor.append(
        set(
            itertools.combinations(physical_axes_by_factor[factor],
                                   multiplicity)))

  for axis_assignment in itertools.product(*assignments_by_factor):
    result = np.ones([len(physical_mesh_shape)], dtype=np.int64)
    for factor_index, per_factor_assignment in enumerate(axis_assignment):
      for physical_axis in per_factor_assignment:
        result[physical_axis] *= factors[factor_index]
    yield result


def _generate_logical_mesh(
    physical_mesh: np.ndarray,
    logical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> np.ndarray:
  """Compute the logical mesh from assignment map.

  Args:
    physical_mesh: Physical device mesh.
    logical_mesh_shape: Logical mesh shape.
    assignment: 2-d assignment matrix shape [physical_dims, logical_dims].

  Returns:
    Logical mesh reshaped from physical mesh.
  """
  physical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(physical_mesh.shape), dtype=np.int64), axis=-1),
      assignment.shape,
  ).reshape([-1])

  logical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(logical_mesh_shape), dtype=np.int64), axis=0),
      assignment.shape,
  ).reshape([-1])

  # Axes of logical mesh is ordered by (physical_axis, logical_axis).
  #
  # Note that we sort for each physical_axis the logical_axis, so that higher
  # intensity logical axes are replicated at inner (minor) dimensions.
  #
  # E.g., if a dimension size is 12 = 3x4, where 3 is higher intensity and 4
  # is lower, we want to reshape so that it becomes 12 = 4x3. Imagine in the
  # 1-d case, this will allow more connections between the higher intensity
  # axes.
  logical_mesh = np.reshape(physical_mesh, assignment.reshape([-1]))

  # We will then group by l_axis as this is what is expected from output.
  _, _, transpose_axes = zip(*sorted(
      zip(logical_indices, physical_indices, range(len(logical_indices)))))
  logical_mesh = np.transpose(logical_mesh,
                              transpose_axes)  # type: ignore  # numpy 2.2

  # Reshape to add the trivial dimensions back.
  logical_mesh = np.reshape(logical_mesh,
                            logical_mesh_shape)  # type: ignore  # numpy 2.2

  return logical_mesh


class MarkShardingFunction(torch.autograd.Function):
  """
  Autograd function to mark_sharding on intermediate tensors and the gradient
  of the intermediate tensors during backward pass.

  Usage:

  >>> new_tensor = MarkShardingFunction.apply(tensor, mesh, ('axis_1', 'axis_2'))

  This is required to guide GSPMD sharding propagation better during the
  backward pass as during complicated workloads the compiler can introduce extra
  collectives that can hurt performance.

  Compared to `mark_sharding`, this version will not in-place shard input tensors.
  Instead it takes in an unsharded tensor and returns a new tensor that is sharded.
  After GSPMD sharding propagation in the compiler, both tensors will become sharded.

  This version can also be used in AOTAutograd.
  """

  @staticmethod
  def forward(ctx, torch_tensor: torch.Tensor, mesh: Mesh,
              partition_spec: PartitionSpec) -> torch.Tensor:
    o = _aot_mark_sharding(torch_tensor, str(mesh), str(partition_spec))
    ctx.partition_spec = partition_spec
    ctx.mesh = mesh
    return o

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):  # type: ignore
    partition_spec = ctx.partition_spec
    mesh = ctx.mesh
    o = _aot_mark_sharding(grad_output, str(mesh), str(partition_spec))
    return o, None, None


@torch.library.custom_op("xla::aot_mark_sharding", mutates_args=())
def _aot_mark_sharding(t: torch.Tensor, mesh: str,
                       partition_spec: str) -> torch.Tensor:
  if t is None:
    return None

  import ast

  import torch_xla.distributed.spmd as xs

  the_mesh = xs.Mesh.from_str(mesh)
  assert the_mesh is not None
  partition_spec_eval = ast.literal_eval(partition_spec)
  return xs.mark_sharding(t.clone(), the_mesh,
                          partition_spec_eval).global_tensor


@_aot_mark_sharding.register_fake
def aot_mark_sharding_fake(t: torch.Tensor, mesh: str,
                           partition_spec: str) -> torch.Tensor:
  if t is None:
    return None
  return torch.empty_like(t)
