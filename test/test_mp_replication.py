import sys
import torch
import torch_xla
import torch_xla_py.xla_model as xm
import torch_xla_py.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  real_device = xm.xla_real_devices([str(device)])[0]
  if real_device.startswith('TPU:'):
    ones = torch.ones((2, 3))
    xones = ones.to(device)
    torch_xla._XLAC._xla_cross_replica_sum([xones], 1.0, [])

    if not xones.cpu().allclose(ones * float(xm.xrt_world_size())):
      print('CrossReplicaSum produced wrong reductions')
      print(xones, file=sys.stderr)
      sys.exit(1)
  else:
    print(
        'Default device {} is not a TPU device'.format(real_device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
