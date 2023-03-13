import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import torch.distributed as dist


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    world_size = xm.rt_world_size()
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


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
