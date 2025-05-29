"""An SPMD program that uses DTensor. There are some gaps but they can be filled."""

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import mark_sharding, Mesh

def main():
  xr.use_spmd()
  device = xm.xla_device()
  world_size = xr.global_runtime_device_count()
  rowwise_placement=[Shard(0)]
  mesh = init_device_mesh("xla", (world_size,))

  t = torch.randn(8, 4, dtype=torch.bfloat16).to(device)
  dt = distribute_tensor(t, mesh, rowwise_placement)
  dt = dt + 2.0
  # The global_tensor is the object that carries the IR trace.
  # The value of 2.0 is broadcast to full size but doesn't get an explicit sharding in the HLO.
  print(torch_xla._XLAC._get_xla_tensors_text([dt.global_tensor]))
  print(torch_xla._XLAC._get_xla_tensors_hlo([dt.global_tensor]))
  print(dt)
  print(dt.global_tensor)
  # This is a list of all four shards
  print(dt.local_shards)

  # This fails because XLAShardedTensor has no attribute 'redistribute'
  # new_dt = dt.redistribute(mesh, [Shard(1)])
  # print(new_dt)

  # We could implement XLAShardedTensor.redistribute and it would
  # basically just apply another mark_sharding, as done below.
  # Strangely it needs an operation to be performed on the sharded tensor,
  # even just multiplying by 1.0 as done here. Otherwise I get the error
  # "Existing annotation must be cleared first", even if acting on global_tensor.
  # The error message doesn't say this but we can clear shardings via
  # torch_xla.distributed.spmd.clear_sharding
  device_ids = np.array(range(world_size))
  mesh_shape = (world_size,)
  xla_mesh = Mesh(device_ids, mesh_shape, ('xla',))
  xla_partition_spec = (None, 0)

  # This is successful, the tensor has a new sharding.
  dt_new = mark_sharding(dt * 1.0, xla_mesh, xla_partition_spec)
  print(dt_new)
  print(dt_new.local_shards)

  # parallelize_module doesn't work atm because the styles (Rowwise/Colwise/Sequence)
  # use redistribute for their inputs/outputs, but we are are planning on overwriting that.
  # On the inputs it also calls DTensor.from_local() if the input is not already a DTensor.
  # We would have to modify that to also not shard if the input is an XLAShardedTensor.
  # The module parameters are sahrded with `distribute_tensor`, so that should work.

  # DTensor.from_local() and DTensor.to_local() don't make sense in this context.

if __name__ == '__main__':
    main()