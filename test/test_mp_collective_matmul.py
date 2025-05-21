import os
import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


def _mp_fn(index):
  os.environ["ENABLE_COLLECTIVE_MATMUL_IN_MP"] = "1"
  device = xm.xla_device()
  world_size = xr.world_size()
  groups = [[i for i in range(world_size)]]
  scale = 1 / world_size
  scatter_dim = 1
  shard_size = 2

  if xm.xla_device_hw(device) in ('TPU',):
    # Testing with a single replica group, channel_id and use_global_device_ids
    ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)
    result = xm.all_gather(
        ordinal_tensor,
        dim=0,
        groups=groups,
        channel_id=1,
        use_global_device_ids=True)
    torch_xla.sync()

    cpu_result = result.cpu()
    expected = torch.arange(0, world_size, dtype=torch.float)
    assert cpu_result.allclose(expected)

    rand = torch.rand((32, shard_size * world_size, 32))
    xrand = rand.to(device)

    res = xm.reduce_scatter(
        xm.REDUCE_SUM,
        xrand,
        scale,
        scatter_dim,
        world_size,
        groups=groups,
        channel_id=1,
        use_global_device_ids=True)
    expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand, scale)
    torch_xla.sync()

    slice_idx = torch.tensor(
        list(range(index * shard_size, (index + 1) * shard_size)))
    expected = expected_world.cpu().index_select(scatter_dim, slice_idx)

    assert res.cpu().allclose(expected)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
