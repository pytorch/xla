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
    rank = xr.global_ordinal()

    dist.init_process_group('xla', init_method='xla://')

    num_all_reduces = 20
    xinputs_list = []
    for i in range(num_all_reduces):
      inputs = torch.ones((2, 3)) * i
      xinputs = inputs.to(device)
      xinputs_list.append(xinputs)
      dist.all_reduce(xinputs)
    torch_xla.sync()
    for i in range(num_all_reduces):
      expected = torch.ones((2, 3)) * i * world_size
      xinputs = xinputs_list[i]
      assert torch.all(
          xinputs.cpu() == expected), f'trial {i}, {xinputs} != {expected}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
