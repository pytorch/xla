"""I was trying to see if operations on sharded tensors result in IR with
explicit collectives. The answer is no."""
import numpy as np
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm


def main():
  xr.use_spmd()
  device = xm.xla_device()

  num_devices = xr.global_runtime_device_count()
  device_ids = np.arange(num_devices)
  mesh_shape = (num_devices,)
  mesh = xs.Mesh(device_ids, mesh_shape, ('data',))

  input_size = 2 * num_devices
  other_dim = num_devices
  x = torch.full((input_size, other_dim), fill_value=1, device=device)
  x_sharded = xs.mark_sharding(x, mesh, partition_spec=(0, None))
  y = torch.transpose(x_sharded, 0, 1)
  y_sharded = xs.mark_sharding(y, mesh, partition_spec=(0, None))

  ir_text = torch_xla._XLAC._get_xla_tensors_text([y_sharded.global_tensor])
  hlo_text = torch_xla._XLAC._get_xla_tensors_hlo([y_sharded.global_tensor])
  dashes = "----------------"
  print(
      f"IR\n{dashes}\n{ir_text}\n\nHLO\n{dashes}\n{hlo_text}\noutput\n{dashes}\n{y_sharded}\n\n"
  )
  torch_xla.sync()


if __name__ == '__main__':
  main()
