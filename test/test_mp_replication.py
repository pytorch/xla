import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) != 'CPU':
    ones = torch.ones((2, 3))
    twos = ones + 1.0
    xones = ones.to(device)
    xtwos = twos.to(device)
    xm.all_reduce('sum', [xones, xtwos])

    if (not xones.cpu().allclose(ones * float(xm.xrt_world_size())) or
        not xtwos.cpu().allclose(twos * float(xm.xrt_world_size()))):
      print('CrossReplicaSum produced wrong reductions', file=sys.stderr)
      print(xones, file=sys.stderr)
      sys.exit(1)
  else:
    print(
        'Default device {} does not support replication'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
