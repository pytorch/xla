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
  if xm.xla_device_hw(device) in ('TPU', 'CUDA'):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    dist.init_process_group('xla', world_size=world_size, rank=rank)

    tensor_list = [
        torch.empty((i, i), device=device) for i in range(1, 1000, 101)
    ]
    for j, t in enumerate(tensor_list):
      t.fill_(float(j))
    dist.bucketed_allreduce(tensor_list)
    for j, t in enumerate(tensor_list):
      assert torch.all(torch.eq(t.cpu(),
                                float(j) * world_size)) == torch.tensor(True)
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
