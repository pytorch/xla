import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU', 'CUDA', 'ROCM', 'NEURON'):
    train_loader = xu.SampleGenerator(
        data=torch.zeros(1, 12), sample_count=1024)
    train_loader = pl.MpDeviceLoader(train_loader, device)
    max_steps = 10
    for step, inputs in enumerate(train_loader):
      xm.all_reduce('sum', [inputs], scale=1.0 / xm.xrt_world_size())
      if step > max_steps:
        break
  else:
    print(f'{device} is not a TPU or GPU device', file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
