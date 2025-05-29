"""Dtensor run in purely multi-process, no SPMD mode, but using the XLA backend."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import init_device_mesh, Shard, DTensor
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

def main(rank: int):
  dist.init_process_group('xla', init_method='xla://')
  world_size = xr.world_size()
  device = xm.xla_device()
  mesh = init_device_mesh("xla", (world_size,))
  # Needed to make DTensor.from_local() work
  mesh._coordinate_on_dim = [rank]

  placements = [Shard(dim=0)]
  local_tensor = torch.full((2, 2), fill_value = rank, dtype=torch.float32, requires_grad=True).to(device)

  dtensor = DTensor.from_local(local_tensor, mesh, placements)
  print(dtensor)
  
  # This fails because of the missing mesh._dim_group_name (see tp_example.py)
  # print(dtensor.full_tensor())

  # This works, it's just the original local_tensor again
  as_local = dtensor.to_local()
  print(as_local)

  # This doesn't work. It dives into DTensor's dispatching logic, which runs a bunch of
  # complicated sharding code to determine how best to move data around.
  # The particular place where it fails is in trying to call torch.cuda.device_count(),
  # replacing ".cuda" with the backend-specific module, but there's no such thing for xla.
  dtensor = dtensor + 2.0


if __name__ == '__main__':
  torch_xla.launch(main, args=(),)