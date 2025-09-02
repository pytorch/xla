import sys
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP


def _mp_fn(index):
  dev = torch_xla.device()
  if xm.xla_device_hw(dev) not in ('TPU',):
    print('Default device {} is not a TPU device'.format(dev), file=sys.stderr)
    return

  model = nn.Linear(1024, 1024)
  model.weight.requires_grad = False  # the weight param is frozen

  model = FSDP(model)  # wrapping the linear module with FSDP

  input = torch.rand((2, 1024), device='xla')

  output = model(input)
  loss = torch.sum(output)
  loss.backward()
  assert not any(p._has_full_param for p in model.full_params), \
    'Expecting all the full params to be freed at this moment.'


if __name__ == "__main__":
  torch_xla.launch(_mp_fn, args=())
