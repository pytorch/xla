import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor

import numpy as np
from typing import Tuple, Union


# torch_xla.distributed.xla_sharding
def mark_sharding(t: torch.Tensor, mesh_shape: Tuple[int],
                  partition_spec: Tuple[Union[int, None]]) -> XLAShardedTensor:
  """
    Annotates the tensor provided with XLA partition spec. Internally,
    it annotates the corresponding XLATensor as sharded for the XLA SpmdPartitioner pass.

    Args:
        t (torch.Tensor): input tensor to be annotated with partition_sepc.

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
  # We might allow {len(partition_spec)} <= {len(t.shape)},
  # where the unspecified ranks are replicated.
  assert len(t.shape) == len(partition_spec), \
    f"Partition spec length ({len(partition_spec)}) is not equal to the input rank ({len(t.shape)})."

  mesh_shape_list = list(mesh_shape)
  partition_spec_list = [-1 if d is None else d for d in list(mesh_shape)]
  torch_xla._XLAC._mark_sharding(t, mesh_shape_list, partition_spec_list)

  return XLAShardedTensor(t)
