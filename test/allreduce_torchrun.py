import torch
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import torch_xla.neuron.distributed.xla_backend
import os

def _mp_fn():
  rank = int(os.environ['RANK'])
  size = int(os.environ['WORLD_SIZE'])

  dist.init_process_group('xla')
  device = xm.xla_device()

  ones = torch.ones((2, 3))
  xones = ones.to(device)
  dist.all_reduce(xones, op=torch.distributed.ReduceOp.SUM)

  result_cpu = xones.cpu()
  expected = torch.ones((2,3))*size
  assert torch.all(xones.cpu() == expected), f'{xones} != {expected}'


if __name__ == '__main__':
    print('master_port:{}, master_addr:{}, rank:{}, local_rank:{}, size:{}'
          .format(os.environ['MASTER_PORT'], os.environ['MASTER_ADDR'], os.environ['RANK'],
                  os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE']))

    _mp_fn()

