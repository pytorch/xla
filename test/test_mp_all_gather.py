import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xm.rt_world_size()
  if xm.xla_device_hw(device) in ('TPU', 'GPU'):
    # Testing with a single replica group
    ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
    result = xm.all_gather(ordinal_tensor, dim=0)

    cpu_result = result.cpu()
    expected = torch.arange(0, world_size, dtype=torch.float)
    if not cpu_result.allclose(expected):
      print('xm.all_gather() produced wrong reductions', file=sys.stderr)
      print(f'[{index}] {cpu_result}', file=sys.stderr)
      sys.exit(1)

    # Testing with two replica groups
    if world_size % 2 == 0 and world_size > 1:
      mp_groups = [[n for n in range(world_size) if n % 2 == 0],
                   [n for n in range(world_size) if n % 2 == 1]]
      group_size = len(mp_groups[0])
      replica_id = int(index % 2 == 1)

      result = xm.all_gather(ordinal_tensor, dim=0, groups=mp_groups)

      cpu_result = result.cpu()
      expected = torch.arange(replica_id, world_size, step=2, dtype=torch.float)
      if not cpu_result.allclose(expected):
        print('xm.all_gather() produced wrong reductions', file=sys.stderr)
        print(f'[{index}] {cpu_result}', file=sys.stderr)
        sys.exit(1)
    else:
      print(
          f'Failed to create two replica groups with {world_size} replicas',
          file=sys.stderr)

  else:
    print(f'{device} is not a TPU or GPU device', file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
