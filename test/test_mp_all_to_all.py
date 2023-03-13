import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) == 'TPU':
    slots_per_device = 4
    size = slots_per_device * xm.rt_world_size()
    ordinal = xm.get_ordinal()
    value = torch.tensor([ordinal] * size, dtype=torch.int32, device=device)
    result_tensor = xm.all_to_all(
        value,
        split_dimension=0,
        concat_dimension=0,
        split_count=xm.rt_world_size())

    result = result_tensor.cpu().tolist()
    for i in range(0, xm.rt_world_size()):
      expected = [i] * slots_per_device
      if expected != result[i * slots_per_device:(i + 1) * slots_per_device]:
        print(
            'Wrong result from core {}: {}'.format(i, result), file=sys.stderr)
        sys.exit(1)
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
