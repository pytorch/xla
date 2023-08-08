import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import torch.distributed as dist


def _test_allgather():
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    input = torch.ones((2, 3)) * rank
    outputs = [torch.zeros_like(input)] * world_size
    xinput = input.to(device)
    xoutputs = [o.to(device) for o in outputs]
    xoutput0 = xoutputs[0]
    dist.all_gather(xoutputs, xinput)
    for i, o in enumerate(xoutputs):
      expected = torch.ones((2, 3)) * i
      assert torch.all(o.cpu() == expected), f'{o} != {expected}'
    expected0 = torch.zeros_like(input)
    assert torch.all(xoutput0.cpu() == expected0), f'{xoutput0} != {expected0}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


def _test__allgather_base():
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    input = torch.ones((2, 3)) * rank
    output = torch.zeros((2 * world_size, 3))
    xinput = input.to(device)
    xoutput = output.to(device)
    dist._all_gather_base(xoutput, xinput)
    xoutputs = torch.split(xoutput, world_size)
    for i, o in enumerate(xoutputs):
      expected = torch.ones((2, 3)) * i
      assert torch.all(o.cpu() == expected), f'{o} != {expected}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    dist.init_process_group('xla', world_size=world_size, rank=rank)
  _test_allgather()
  _test__allgather_base()


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
