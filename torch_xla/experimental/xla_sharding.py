from collections import OrderedDict
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.pjrt import requires_pjrt

import numpy as np
from typing import Tuple, Union, List


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
  —------------------------------
  mesh_shape = (4, 2)
  num_devices = len(xm.get_xla_supported_devices())
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
  mesh.get_logical_mesh()
  >> array([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7]])
  mesh.shape()
  >> OrderedDict([('x', 4), ('y', 2)])
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
    assert (len(device_ids) == np.prod(mesh_shape))
    assert len(device_ids) == len(np.unique(device_ids))
    self.device_ids = device_ids
    self.mesh_shape = mesh_shape
    self.axis_names = axis_names
    assert all(d < self.size() for d in device_ids)

  def size(self):
    return np.prod(self.mesh_shape)

  def shape(self):
    return OrderedDict(
        (name, size) for name, size in zip(self.axis_name, self.mesh_shape))

  def get_logical_mesh(self):
    return self.device_ids.reshape(self.mesh_shape)


@requires_pjrt
def mark_sharding(t: Union[torch.Tensor, XLAShardedTensor], mesh: Mesh,
                  partition_spec: Tuple[Union[int, None]]) -> XLAShardedTensor:
  """
    Annotates the tensor provided with XLA partition spec. Internally,
    it annotates the corresponding XLATensor as sharded for the XLA SpmdPartitioner pass.
    Args:
        t (Union[torch.Tensor, XLAShardedTensor]): input tensor to be annotated with partition_sepc.

        mesh (Mesh): describes the logical XLA device topology and the underlying device IDs.

        partition_spec (Tuple[int, None]): A tuple of device_mesh dimension index or `None`.
        This specifies how each input rank is sharded (index to mesh_shape) or replicated (None).
        For example, we can shard an 8x10 tensor 4-way row-wise, and replicate column-wise.
        >> input = torch.randn(8, 10)
        >> mesh_shape = (4, 2)
        >> partition_spec = (0, None)

    Examples
    —------------------------------
    mesh_shape = (4, 2)
    num_devices = len(xm.get_xla_supported_devices())
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

    # 4-way data parallel
    input = torch.randn(8, 32).to(xm.xla_device())
    xs.mark_sharding(input, mesh, (0, None))

    # 2-way model parallel
    linear = nn.Linear(32, 10).to(xm.xla_device())
    xs.mark_sharding(linear.weight, mesh, (None, 1))
  """
  num_devices = len(xm.get_xla_supported_devices())
  assert num_devices > 0, "This requires XLA supported device(s)."
  assert mesh.size() == num_devices, \
    f"{mesh.mesh_shape} is not mappable over {num_devices} devices."
  assert all((d >= 0 and d < len(mesh.mesh_shape)) for d in partition_spec if d), \
    f"partition_spec ({partition_spec}) contains out of bound index into mesh_shape."
  # TODO(yeounoh) allow unspecified ranks (len(partition_spec) <= len(t.shape)),
  # for replication. For now, all input rank sharding should be specified.
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) is not equal to the input rank ({len(t.shape)})."
  dims = [d for d in partition_spec if d]
  assert len(dims) == len(np.unique(dims)), \
    f"Each device mesh dimension should appear at most once in partition_spec {partition_spec}."

  tile_assignment = mesh.get_logical_mesh().tolist()
  manual, replicated, partial = False, False, False
  if all(d is None for d in partition_spec):
    replicated = True
  elif any(d is None for d in partition_spec):
    partial = True
  # TODO(yeounoh) suport partially replicated sharding.
  assert not partial, "Partial replication is currently not supported."

  if isinstance(t, XLAShardedTensor):
    torch_xla._XLAC._xla_mark_sharding(t.global_tensor, tile_assignment,
                                       replicated, manual)
    return t
  torch_xla._XLAC._xla_mark_sharding(t, tile_assignment, replicated, manual)
  return XLAShardedTensor(t)


def clear_sharding(t: Union[torch.Tensor, XLAShardedTensor]) -> torch.Tensor:
  """Clear sharding annotation from the input tensor and return a `cpu` casted tensor."""
  torch_xla._XLAC._xla_clear_sharding(t)
  if isinstance(t, XLAShardedTensor):
    return t.global_tensor
  return t
