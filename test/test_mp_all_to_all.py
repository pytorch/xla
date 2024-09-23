import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'NEURON'):
    slots_per_device = 4
    size = slots_per_device * xr.world_size()
    ordinal = xr.global_ordinal()
    value = torch.tensor([ordinal] * size, dtype=torch.int32, device=device)
    result_tensor = xm.all_to_all(
        value,
        split_dimension=0,
        concat_dimension=0,
        split_count=xr.world_size())

    result = result_tensor.cpu().tolist()
    for i in range(0, xr.world_size()):
      expected = [i] * slots_per_device
      if expected != result[i * slots_per_device:(i + 1) * slots_per_device]:
        print(
            'Wrong result from core {}: {}'.format(i, result), file=sys.stderr)
        sys.exit(1)
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
