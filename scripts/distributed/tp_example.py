"""A native, multi-process PT program that uses DTensor, run with the XLA backend.
The rub is that as soon as `distribute_tensor()` gets called, `xr.use_spmd()` is invoked.
Which means we are running in SPMD mode even though there are multiple processes.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import init_device_mesh, Shard, DTensor, distribute_tensor
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm


def main(rank: int):
  dist.init_process_group("xla", init_method='xla://')
  world_size = xr.world_size()
  device = xm.xla_device()
  mesh = init_device_mesh("xla", (world_size,))

  # Attempts to patch from_local
  mesh._coordinate_on_dim = [rank]
  mesh._dim_group_names = ["xla"]

  # Basic use of distribute_tensor DOES work
  x = torch.ones(8, 4).to(device)
  y = torch.ones(8, 4).to(device)
  dt_x = distribute_tensor(x, mesh, [Shard(dim=0)])
  dt_y= distribute_tensor(y, mesh, [Shard(dim=0)])

  # This doesn't work if it's dt_x + 2.0, indicating significant problems using dtensor in multi-process mode.
  # (Specifically it's the print statement that doesn't work)
  dt_z = dt_x + dt_y
  print(torch_xla._XLAC._get_xla_tensors_text([dt_z.global_tensor]))
  print(torch_xla._XLAC._get_xla_tensors_hlo([dt_z.global_tensor]))
  print(dt_z)

  # This does work and could be used for to_local(),
  # but it makes use of global_tensor, which is a copy
  # of the whole tensor stored in CPU. I guess this makes sense
  # in SPMD mode, where the tensor isn't actually materialized.
  # But it feels like poor form in MPMD mode.
  print(dt_z.local_shards)

  # This fails because there's no XLAShardedTensor.placements attribute,
  # but it could be easily added and would read from the partition spec
  # print(my_dtensor.placements())

  # from_local succeeds, but the call to full_tensor fails because there's no mesh._coordinate_on_dim
  # That should be [rank] (or whatever the coordinates are for an N-d mesh).
  # But then it fails because there's no mesh._dim_group_names.
  # These seem to relate to the name of the collective communications
  # process group, of which there are a fixed set (see torch/distributed/distributed_c10d.py):
  # gloo, nccl, etc. Based on xla/torch_xla/distributed/xla_backend.py it looks
  # like we registered a process group backend with the name 'xla', but there's
  # also a comment about not implementing the C++/Python extension.
  # Whatever the case, setting `mesh._dim_group_name = ['xla']` yields the error
  # "Could not resolve the process group registered under the name xla".
  # Note that this code path is also exercised by DTensor.redistribute()

  # rowwise_placement=[Shard(0)]
  # local_tensor = torch.full((2, 4), fill_value = rank, dtype=torch.float32, requires_grad=True)
  # rowwise_tensor = DTensor.from_local(local_tensor, mesh, rowwise_placement)

  # rowwise_tensor = rowwise_tensor + 2.0
  # print(rowwise_tensor.to_local())

  # full_tensor = rowwise_tensor.full_tensor()
  # print(full_tensor)

if __name__ == '__main__':
  torch_xla.launch(main, args=())
