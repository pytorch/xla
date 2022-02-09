import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
    result = xm.all_gather(ordinal_tensor)

    cpu_result = result.cpu()
    expected = torch.arange(0, world_size, dtype=torch.float)
    if not cpu_result.allclose(expected):
      print('xm.all_gather() produced wrong reductions', file=sys.stderr)
      print('[{}] {}'.format(index, cpu_result), file=sys.stderr)
      sys.exit(1)
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
