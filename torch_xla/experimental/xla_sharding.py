import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor

import numpy as np
from typing import Tuple, Union


def mark_sharding(t: Union[torch.Tensor,
                           XLAShardedTensor], mesh_shape: Tuple[int],
                  partition_spec: Tuple[Union[int, None]]) -> XLAShardedTensor:
  """
    Annotates the tensor provided with XLA partition spec. Internally,
    it annotates the corresponding XLATensor as sharded for the XLA SpmdPartitioner pass.
    Args:
        t (Union[torch.Tensor, XLAShardedTensor]): input tensor to be annotated with partition_sepc.
        mesh_shape (Tuple[Union[int, None]]): A int tuple describing the logical topology
        of the device mesh, and each element describes the number of devices in
        the corresponding axis.
        partition_spec (Tuple[int, None]): A tuple of device_mesh dimension index or `None`.
        This specifies how each input rank is sharded (index to mesh_shape) or replicated (None).
        For example, we can shard an 8x10 tensor 4-way row-wise, and replicate column-wise.
        >> input = torch.randn(8, 10)
        >> mesh_shape = (4, 2)
        >> assert np.prod(mesh_shape) == xm.xrt_world_size()
        >> partition_spec = (0, None)
        >> assert len(input.shape) == len(partition_spec)
    Examples
    —------------------------------
    mesh_shape = (4, 2)
    input = torch.randn(8, 32).to(xm.xla_device())
    # 4-way data parallel
    input = xs.mark_sharding(input, mesh_shape, (0, None))
    linear = nn.Linear(32, 10).to(xm.xla_device())
    # 2-way model parallel
    linear.weight = xs.mark_sharding(linear.weight, device_mesh, (None, 1))
    output = linear(input)
    # full replication
    output = xs.mark_sharding(output, device_mesh, (None, None))
  """
  num_devices = xm.xrt_world_size()
  assert np.prod(mesh_shape) == num_devices, \
    f"{mesh_shape} is not mappable over {num_devices} devices."
  assert all((d >= 0 and d < len(mesh_shape)) for d in partition_spec if d), \
    f"partition_spec ({partition_spec}) contains out of bound index into mesh_shape."
  # TODO(yeounoh) allow unspecified ranks (len(partition_spec) <= len(t.shape)),
  # for replication. For now, all input rank sharding should be specified.
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) is not equal to the input rank ({len(t.shape)})."
  dims = [d for d in partition_spec if d]
  assert len(dims) == len(np.unique(dims)), \
    f"Each device mesh dimension should appear at most once in partition_spec {partition_spec}."

  device_ids = np.array(range(num_devices))
  tile_assignment = device_ids.reshape(mesh_shape).tolist()

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
  return NotImplemented
