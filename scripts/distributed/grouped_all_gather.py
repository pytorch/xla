"""Try running an all_gather operation on a limited process group."""

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def main(rank: int):
  dist.init_process_group("xla", init_method='xla://')
  device = xm.xla_device()
  world_size = xr.world_size()
  input_size = 2
  a = torch.full(
      size=(input_size,), fill_value=rank, device=device, dtype=torch.bfloat16)

  group_size = 2
  group1_ranks = [0, 1]
  group2_ranks = [2, 3]
  group = dist.new_group(ranks=group1_ranks)
  output_tensors_list = [
      torch.zeros((input_size,), dtype=torch.bfloat16, device=device)
      for _ in range(group_size)
  ]

  dist.all_gather(tensor_list=output_tensors_list, tensor=a, group=group)

  print(f"RANK {rank}: {output_tensors_list}\n")
  xm.mark_step()


if __name__ == '__main__':
  torch_xla.launch(main, args=())
