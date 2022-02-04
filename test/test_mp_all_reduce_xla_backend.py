import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.distributed as dist


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    print(f'RANK {rank} init\'ing torch.distributed...')
    dist.init_process_group('xla', world_size=world_size, rank=rank)
    print(f'RANK {rank} init\'ed torch.distributed.')

    ones = torch.ones((2, 3))
    print("transferring ones to device")
    xones = ones.to(device)
    print("running all reduce")
    dist.all_reduce(xones)
    expected = torch.ones((2, 3)) * world_size
    assert torch.all(xones.cpu() == expected), f'{xones} != {expected}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
