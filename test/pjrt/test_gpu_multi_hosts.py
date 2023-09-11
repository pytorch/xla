import logging
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.experimental.pjrt_backend
from torch_xla._internal.pjrt import *
from torch_xla._internal import gpu, pjrt
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
from torch_xla.distributed.fsdp.wrap import (
    always_wrap_policy as always_wrap,)
from multiprocessing import Process


class MyLinear(nn.Linear):
  """Linear layer with deterministic reset_parameters for testing."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def reset_parameters(self, *args, **kwargs):
    with torch.no_grad():
      self.weight.fill_(1)


class MyModel(nn.Module):

  def __init__(self, device):
    super().__init__()
    self.lin1 = MyLinear(2, 2, bias=False, device=device)
    self.lin2 = MyLinear(2, 2, bias=False, device=device)

  def forward(self, x):
    return self.lin2(self.lin1(x))

  def reset_parameters(self, *args, **kwargs):
    for m in [self.lin1, self.lin2]:
      if not isinstance(m, XlaFullyShardedDataParallel):
        m.reset_parameters()


def forward():
  with torch.no_grad():
    device = xm.xla_device()
    model = MyModel(device)
    inp = torch.randn(10, 2, device=device)
    logits = model(inp)
    xm.mark_step()
    return logits


def _mp_fn(index, *args, **kwargs):
  dist.init_process_group('xla', init_method='xla://')
  logits = forward()
  output_tensors = [
      torch.zeros_like(logits, device=xm.xla_device())
      for _ in range(int(os.environ['PJRT_WORLD_SIZE']))
  ]
  # test all-gather
  dist.all_gather(output_tensors, logits)
  logits_cpu = [output_tensor.to('cpu') for output_tensor in output_tensors]
  # test all-reduce
  dist.all_reduce(logits)
  return logits_cpu


def worker_fn(local_rank,
              group_rank,
              local_world_size,
              world_size,
              cuda_visible_devices='',
              *args,
              **kwargs):
  os.environ[xenv.PJRT_LOCAL_RANK] = str(local_rank)
  os.environ[xenv.PJRT_LOCAL_WORLD_SIZE] = str(local_world_size)
  os.environ[xenv.PJRT_GROUP_RANK] = str(group_rank)
  os.environ[xenv.PJRT_RANK] = str(local_rank + group_rank * local_world_size)
  os.environ[xenv.PJRT_WORLD_SIZE] = str(world_size)
  os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
  pjrt.initialize_multiprocess(local_rank, group_rank, local_world_size,
                               world_size)
  return _mp_fn(local_rank, *args, **kwargs)


def master_fn(world_size):
  gpu.shutdown_distributed_runtime()
  gpu.initialize_distributed_runtime(world_size)
  time.sleep(3600)


def test_spawn_single_hosts():

  def _fn():
    WORLD_SIZE = 2
    os.environ['GPU_NUM_DEVICES'] = str(WORLD_SIZE)
    os.environ.setdefault('PJRT_WORLD_SIZE', str(WORLD_SIZE))
    os.environ.setdefault('PJRT_GROUP_RANK', str(0))
    # test multiprocessing
    xmp.spawn(_mp_fn, args=())

  p = Process(target=_fn, args=())
  p.start()
  p.join()


def test_torchelastic_launch_multi_hosts():
  world_size = 2
  master_proc = Process(target=master_fn, args=(world_size,))
  master_proc.start()
  os.environ['GPU_NUM_DEVICES'] = str(world_size)
  os.environ['PJRT_WORLD_SIZE'] = str(world_size)
  configs = (
      ((0, 0, 1, world_size), {
          'cuda_visible_devices': '0'
      }),
      ((0, 1, 1, world_size), {
          'cuda_visible_devices': '1'
      }),
  )
  procs = []
  for config in configs:
    procs.append(Process(target=worker_fn, args=config[0], kwargs=config[1]))
    procs[-1].start()
  for p in procs:
    p.join()
  master_proc.kill()


def test_torchelastic_launch_single_host():
  world_size = 2
  master_proc = Process(target=master_fn, args=(world_size,))
  master_proc.start()
  os.environ['GPU_NUM_DEVICES'] = str(world_size)
  os.environ['PJRT_WORLD_SIZE'] = str(world_size)
  configs = (
      ((0, 0, 2, world_size), {
          'cuda_visible_devices': '0,1'
      }),
      ((1, 0, 2, world_size), {
          'cuda_visible_devices': '0,1'
      }),
  )
  procs = []
  for config in configs:
    procs.append(Process(target=worker_fn, args=config[0], kwargs=config[1]))
    procs[-1].start()
  for p in procs:
    p.join()
  master_proc.kill()


def test_torchelastic_launch_two_procs_per_host():
  world_size = 8
  master_proc = Process(target=master_fn, args=(world_size,))
  master_proc.start()
  os.environ['GPU_NUM_DEVICES'] = str(world_size)
  os.environ['PJRT_WORLD_SIZE'] = str(world_size)
  configs = (
      ((0, 0, 2, world_size), {
          'cuda_visible_devices': '0,1'
      }),
      ((1, 0, 2, world_size), {
          'cuda_visible_devices': '0,1'
      }),
      ((0, 1, 2, world_size), {
          'cuda_visible_devices': '2,3'
      }),
      ((1, 1, 2, world_size), {
          'cuda_visible_devices': '2,3'
      }),
      ((0, 2, 2, world_size), {
          'cuda_visible_devices': '4,5'
      }),
      ((1, 2, 2, world_size), {
          'cuda_visible_devices': '4,5'
      }),
      ((0, 3, 2, world_size), {
          'cuda_visible_devices': '6,7'
      }),
      ((1, 3, 2, world_size), {
          'cuda_visible_devices': '6,7'
      }),
  )
  procs = []
  for config in configs:
    procs.append(Process(target=worker_fn, args=config[0], kwargs=config[1]))
    procs[-1].start()
  for p in procs:
    p.join()
  master_proc.kill()


if __name__ == '__main__':
  num_devices = int(os.environ.get(xenv.GPU_NUM_DEVICES, '0'))
  pjrt_device = os.environ.get(xenv.PJRT_DEVICE, '')
  os.environ.setdefault('PJRT_DIST_SERVICE_ADDR', '127.0.0.1:30285')
  if pjrt_device == 'GPU' and num_devices > 1:
    test_spawn_single_hosts()
    test_torchelastic_launch_multi_hosts()
    test_torchelastic_launch_single_host()
    if num_devices >= 8:
      test_torchelastic_launch_two_procs_per_host()
  else:
    logging.warning(
        'Skip PJRT GPU multiple hosts test because number of devices (%s) <= 1',
        num_devices)
