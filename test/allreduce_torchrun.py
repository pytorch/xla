import argparse
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.distributed.xrt_init import init_xrt_context
import torch_xla.distributed.xla_backend


def _mp_fn_xrt_init():
  rank = int(os.environ['RANK'])
  size = int(os.environ['WORLD_SIZE'])

  init_xrt_context()

  device = xm.xla_device()
  ones = torch.ones((2, 3))
  xones = ones.to(device)
  result = xm.all_reduce('sum', xones)

  result_cpu = result.cpu()
  expected = torch.ones((2,3)) * size
  assert torch.all(result_cpu == expected), f'{result_cpu} != {expected}'


def _mp_fn_xla_backend():
  rank = int(os.environ['RANK'])
  size = int(os.environ['WORLD_SIZE'])

  dist.init_process_group('xla')
  device = xm.xla_device()

  ones = torch.ones((2, 3))
  xones = ones.to(device)
  dist.all_reduce(xones, op=torch.distributed.ReduceOp.SUM)

  result_cpu = xones.cpu()
  expected = torch.ones((2,3)) * size
  assert torch.all(xones.cpu() == expected), f'{xones} != {expected}'


if __name__ == '__main__':
  print('master_port:{}, master_addr:{}, rank:{}, local_rank:{}, size:{}'
        .format(os.environ['MASTER_PORT'], os.environ['MASTER_ADDR'], os.environ['RANK'],
                os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE']))
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_xla_backend', action="store_true")
  args = parser.parse_args()
  if args.use_xla_backend:
    _mp_fn_xla_backend()
  else:
    _mp_fn_xrt_init()
