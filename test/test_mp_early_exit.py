from absl import logging
import sys
import torch
import torch.distributed as dist
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.utils.utils as xu


def _mp_fn():
  dist.init_process_group('xla', init_method='xla://')
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ['TPU', 'CUDA']:
    train_loader = xu.SampleGenerator(
        data=torch.zeros(1, 12), sample_count=1024)
    train_loader = pl.MpDeviceLoader(train_loader, device)
    max_steps = 10
    for step, inputs in enumerate(train_loader):
      xm.all_reduce('sum', [inputs], scale=1.0 / xr.world_size())
      if step > max_steps:
        break
  else:
    print(f'{device} is not a TPU or GPU device', file=sys.stderr)


if __name__ == '__main__':
  if not dist.is_torchelastic_launched():
    logging.error('Test must be launched with torchrun!')
    exit(1)
  _mp_fn()
