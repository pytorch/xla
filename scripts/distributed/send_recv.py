"""Script used to try and get send/recv working. I concluded that the corresponding
XLA ops require "frontend attributes" containing the full set of source-sink pairs,
which is not avilable in the PT Native "one independent process per device" paradigm.
Instead, we need to use xm.collective_permute()"""
import torch
import torch_xla
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr


def main(rank: int):
  dist.init_process_group("xla", init_method="xla://")
  device = xm.xla_device()
  input_tensor_val = float(xr.global_ordinal()) + 1.0
  input_size = 4
  input_tensor = torch.full((input_size,),
                            input_tensor_val,
                            dtype=torch.float32,
                            device=device)
  print(f"rank {rank} input = {input_tensor}")

  # output_tensor = torch.zeros_like(input_tensor, device=device)

  # if rank < midpoint:
  #   xm.send(input_tensor, channel_id=rank)
  # else:
  #   xm.recv(output_tensor, channel_id=rank - midpoint)

  # dist.send(input_tensor, dst=(rank + 1) % xr.world_size())
  # dist.recv(output_tensor, src=(rank - 1) % xr.world_size())

  # pairs = [[i, (i + 1) % world_size] for i in range(world_size)]
  pairs = [[0, 2], [1, 3]]
  def callable(t: torch.Tensor):
    return xm.collective_permute(t, pairs)
  compiled = torch.compile(callable, backend='openxla', fullgraph=False)
  output_tensor = compiled(input_tensor)
  print(f"rank {rank} output = {output_tensor}")
  dist.destroy_process_group()


if __name__ == '__main__':
  torch_xla.launch(main, args=())
