"""Print the IR and HLO for an all_gather operation."""

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def main(rank: int):
  dist.init_process_group("xla", init_method='xla://')

  device = xm.xla_device()
  input_size = 2
  val = float(xr.global_ordinal())
  a = torch.full(
      size=(input_size,), fill_value=val, device=device, dtype=torch.bfloat16)

  output_tensors_list = [
      torch.zeros((input_size,), dtype=torch.bfloat16, device=device)
      for _ in range(xr.world_size())
  ]

  dist.all_gather(tensor_list=output_tensors_list, tensor=a)

  ir_text = torch_xla._XLAC._get_xla_tensors_text(output_tensors_list)
  hlo_text = torch_xla._XLAC._get_xla_tensors_hlo(output_tensors_list)
  dashes = "----------------"
  if rank == 0 or rank == xr.world_size() - 1:
    print(
        f"RANK {rank}\n{dashes}\n\nIR\n{dashes}\n{ir_text}\n\nHLO\n{dashes}\n{hlo_text}\noutput\n{dashes}\n{output_tensors_list}\n\n"
    )
  xm.mark_step()


if __name__ == '__main__':
  torch_xla.launch(main, args=())
