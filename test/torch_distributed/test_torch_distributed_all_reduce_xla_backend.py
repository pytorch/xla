import os
import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch.distributed as dist


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'CUDA', 'NEURON'):
    world_size = xr.world_size()
    dist.init_process_group('xla', init_method='xla://')
    # note that we can't use torch.tensor(torch.distributed.get_rank()) directly
    # since 0 and 1 will be special case into constant. In collective ops we need
    # can't have some of the tensors becomes constant while others are device data.
    rank_tensor = torch.tensor([torch.distributed.get_rank()])
    xla_rank_tensor = rank_tensor.to(device)
    dist.all_reduce(xla_rank_tensor)
    expected = torch.tensor([(world_size - 1) * world_size / 2])
    torch_xla.sync()
    assert torch.all(
        xla_rank_tensor.cpu() == expected), f'{xla_rank_tensor} != {expected}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
