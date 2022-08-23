import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import torch.distributed as dist


def _test_reduce_scatter():
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    dist.init_process_group('xla', world_size=world_size, rank=rank)

    input_size = (32, 3)
    inputs = torch.ones(input_size).split(input_size[0] // world_size)
    output = torch.zeros_like(inputs[0])
    xinputs = [i.to(device) for i in inputs]
    xoutput = output.to(device)
    dist.reduce_scatter(xoutput, xinputs)
    expected = torch.ones_like(inputs[0]) * world_size
    assert torch.all(xoutput.cpu() == expected), f'{xoutput} != {expected}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


def _test__reduce_scatter_base():
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    dist.init_process_group('xla', world_size=world_size, rank=rank)

    input_size = (32, 3)
    input = torch.ones(input_size)
    output = torch.zeros((input_size[0] // world_size, input_size[1]))
    xinput = input.to(device)
    xoutput = output.to(device)
    dist._reduce_scatter_base(xoutput, xinput)
    expected = torch.ones_like(output) * world_size
    assert torch.all(xoutput.cpu() == expected), f'{xoutput} != {expected}'
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


def _mp_fn(index):
  _test_reduce_scatter()
  _test__reduce_scatter_base()


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
