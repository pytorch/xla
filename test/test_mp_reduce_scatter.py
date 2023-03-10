import sys
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xm.rt_world_size()
  scale = 1 / world_size
  scatter_dim = 1
  shard_size = 2

  if xm.xla_device_hw(device) == 'TPU':
    rand = torch.rand((32, shard_size * world_size, 32))
    xrand = rand.to(device)

    res = xm.reduce_scatter(xm.REDUCE_SUM, xrand, scale, scatter_dim,
                            world_size)
    expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand, scale)
    xm.mark_step()

    slice_idx = torch.tensor(
        list(range(index * shard_size, (index + 1) * shard_size)))
    expected = expected_world.cpu().index_select(scatter_dim, slice_idx)

    assert res.cpu().allclose(expected)

    xm.rendezvous('test_reduce_scatter')
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
