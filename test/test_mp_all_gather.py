import os
import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


def all_gather(tensor, dim):
  return xm.all_gather(tensor, dim=dim)


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  input_list_size = 5
  if xm.xla_device_hw(device) in ('TPU', 'CUDA', 'NEURON'):
    # Testing with a single replica group
    ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
    result = xm.all_gather(ordinal_tensor, dim=0)
    torch_xla.sync()

    cpu_result = result.cpu()
    expected = torch.arange(0, world_size, dtype=torch.float)
    if not cpu_result.allclose(expected):
      print('xm.all_gather() produced wrong reductions', file=sys.stderr)
      print(f'[{index}] {cpu_result}', file=sys.stderr)
      sys.exit(1)

    compiled_all_gather = torch.compile(
        all_gather, backend='openxla', fullgraph=True)
    ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
    result = compiled_all_gather(ordinal_tensor, dim=0)

    cpu_result = result.cpu()
    expected = torch.arange(0, world_size, dtype=torch.float)
    if not cpu_result.allclose(expected):
      print(
          'xm.all_gather() produced wrong reductions (torch.compile)',
          file=sys.stderr)
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

    # Testing with a single replica group and tensor list as input
    ordinal_tensors = [
        torch.tensor([i * 1000 + index], dtype=torch.float).to(device)
        for i in range(input_list_size)
    ]
    # TODO: add support for list input with pin_layout=True and output=None
    result_list = xm.all_gather(ordinal_tensors, dim=0, pin_layout=False)

    for i, result in enumerate(result_list):
      cpu_result = result.cpu()
      expected = i * 1000 + torch.arange(world_size, dtype=torch.float)
      if not cpu_result.allclose(expected):
        print(
            'xm.all_gather() produced wrong reductions for item {i} in result list',
            file=sys.stderr)
        print(f'[{index}] {cpu_result}', file=sys.stderr)
        sys.exit(1)

    # Testing with a single replica group and tensor list as input and output!=None (out-of-place)
    # Reuse ordinal_tensors from previous test
    output_tensors = [
        torch.zeros([world_size], dtype=torch.float).to(device)
        for i in range(input_list_size)
    ]
    # TODO: add support for list input with pin_layout=True and output!=None
    result_list = xm.all_gather(
        ordinal_tensors, dim=0, output=output_tensors, pin_layout=False)

    for i, result in enumerate(result_list):
      cpu_result = result.cpu()
      expected = i * 1000 + torch.arange(world_size, dtype=torch.float)
      if not cpu_result.allclose(expected):
        print(
            'xm.all_gather() produced wrong reductions for item {i} in result list',
            file=sys.stderr)
        print(f'[{index}] {cpu_result}', file=sys.stderr)
        sys.exit(1)

    # Testing with a single replica group and tensor list as input (Bucketized)
    # TODO: add support for list input with pin_layout=True and output=None
    result_list = xm.all_gather_bucketized(
        ordinal_tensors, dim=0, pin_layout=False)

    for i, result in enumerate(result_list):
      cpu_result = result.cpu()
      expected = i * 1000 + torch.arange(world_size, dtype=torch.float)
      if not cpu_result.allclose(expected):
        print(
            'xm.all_gather_bucketized() produced wrong reductions for item {i} in result list',
            file=sys.stderr)
        print(f'[{index}] {cpu_result}', file=sys.stderr)
        sys.exit(1)

    # Testing with a single replica group and tensor list as input and output!=None (out-of-place) (Bucketized)
    # Reuse ordinal_tensors from previous test
    output_tensors = [
        torch.zeros([world_size], dtype=torch.float).to(device)
        for i in range(input_list_size)
    ]
    # TODO: add support for list input with pin_layout=True and output!=None
    result_list = xm.all_gather_bucketized(
        ordinal_tensors, dim=0, output=output_tensors, pin_layout=False)

    for i, result in enumerate(result_list):
      cpu_result = result.cpu()
      expected = i * 1000 + torch.arange(world_size, dtype=torch.float)
      if not cpu_result.allclose(expected):
        print(
            'xm.all_gather() produced wrong reductions for item {i} in result list',
            file=sys.stderr)
        print(f'[{index}] {cpu_result}', file=sys.stderr)
        sys.exit(1)

    # Testing with a single replica group and tensor list as input and output!=None (out-of-place) (Bucketized, zero bucket size)
    # Reuse ordinal_tensors from previous test
    output_tensors = [
        torch.zeros([world_size], dtype=torch.float).to(device)
        for i in range(input_list_size)
    ]
    # TODO: add support for list input with pin_layout=True and output!=None
    result_list = xm.all_gather_bucketized(
        ordinal_tensors,
        dim=0,
        output=output_tensors,
        pin_layout=False,
        bucket_cap_mb=0)

    for i, result in enumerate(result_list):
      cpu_result = result.cpu()
      expected = i * 1000 + torch.arange(world_size, dtype=torch.float)
      if not cpu_result.allclose(expected):
        print(
            'xm.all_gather() produced wrong reductions for item {i} in result list',
            file=sys.stderr)
        print(f'[{index}] {cpu_result}', file=sys.stderr)
        sys.exit(1)

    # TODO: add test for torch.compile when support for list input is ready

  else:
    print(f'{device} is not a TPU or GPU device', file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
