import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) == 'TPU':
    world_size = xm.xrt_world_size()
    ordinal = xm.get_ordinal()
    value = torch.tensor([ordinal] * 100, dtype=torch.int32, device=device)
    pairs = []
    for i in range(1, world_size):
      pairs.append([i - 1, i])
    pairs.append([world_size - 1, 0])
    result_tensor = xm.collective_permute(value, pairs)

    result = result_tensor.cpu().tolist()
    expected = [ordinal - 1] * 100 if ordinal != 0 else [world_size - 1] * 100

    if result != expected:
      print(
          'Wrong result from core {}: {}'.format(ordinal, result),
          file=sys.stderr)
      sys.exit(1)
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
