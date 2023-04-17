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
    xones = ones.to(device)
    xtwos = twos.to(device)
    # xm.all_reduce(xm.REDUCE_SUM, [xones, xtwos])
    # xones = all_reduce(xones)
    # xtwos = all_reduce(xtwos)
    # xones = torch._C._nn.all_reduce(xones, xm.REDUCE_SUM, "", [], 0)
    # xtwos = torch._C._nn.all_reduce(xtwos, xm.REDUCE_SUM, "", [], 0)

    compiled_all_reduce = torch.compile(all_reduce, backend='torchxla_trace_once', fullgraph=True)
    xones = compiled_all_reduce(xones)
    xtwos = compiled_all_reduce(xtwos)

    if (not xones.cpu().allclose(ones * float(world_size)) or
        not xtwos.cpu().allclose(twos * float(world_size))):
      print('xm.all_reduce() produced wrong reductions', file=sys.stderr)
      print(xones, file=sys.stderr)
      sys.exit(1)

    xm.master_print(xones)
    xm.master_print(xtwos)
  else:
    print(
        'Default device {} does not support replication'.format(device),
        file=sys.stderr)

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
