import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


def dummy_collective_fn(input):
  res_tensor = xm.all_reduce(xm.REDUCE_SUM, input)
  res_tensor += 3.0
  res_tensor = xm.all_gather(res_tensor, dim=0)
  return res_tensor


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  if xm.xla_device_hw(device) not in ('TPU', 'CUDA', 'NEURON'):
    print(f'skip this test for hw {xm.xla_device_hw(device)}')
  ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
  for dynamic in [True, False]:
    met.clear_all()
    compiled_collective = torch.compile(
        dummy_collective_fn, backend="openxla", dynamic=dynamic)
    res_tensor = compiled_collective(ordinal_tensor)
    expected_tensor = torch.tensor(
        [world_size * world_size / 2] * world_size, dtype=torch.float) + 3.0
    torch_xla.sync()
    torch.allclose(res_tensor.cpu(), expected_tensor)
    assert met.metric_data("ExecuteTime")[0] == 1


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
