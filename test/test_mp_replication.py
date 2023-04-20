import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def all_reduce(tensor):
  return xm.all_reduce(xm.REDUCE_SUM, tensor)

def _mp_fn(index):
  device = xm.xla_device()
  world_size = xm.xrt_world_size()
  if world_size > 1:
    ones = torch.ones((2, 3))
    twos = ones + 1.0
    threes = ones + 2.0
    fours = ones + 3.0
    fives = ones + 4.0
    scale = 0.5
    xones = ones.to(device)
    xtwos = twos.to(device)
    xthrees = threes.to(device)
    xfours = fours.to(device)
    xfives = fives.to(device)
    xm.all_reduce(xm.REDUCE_SUM, [xones, xtwos])
    xthrees = all_reduce(xthrees)
    xfours = xm.all_reduce(xm.REDUCE_SUM, xfours, scale=scale)

    compiled_all_reduce = torch.compile(all_reduce, backend='torchxla_trace_once', fullgraph=True)
    xfives = compiled_all_reduce(xfives)

    if (not xones.cpu().allclose(ones * float(world_size)) or
        not xtwos.cpu().allclose(twos * float(world_size)) or
        not xthrees.cpu().allclose(threes * float(world_size)) or
        not xfours.cpu().allclose(fours * float(world_size) * scale) or
        not xfives.cpu().allclose(fives * float(world_size))):
      print('xm.all_reduce() produced wrong reductions', file=sys.stderr)
      print(xones, file=sys.stderr)
      print(xtwos, file=sys.stderr)
      print(xthrees, file=sys.stderr)
      print(xfours, file=sys.stderr)
      print(xfives, file=sys.stderr)
      sys.exit(1)
  else:
    print(
        'Default device {} does not support replication'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
