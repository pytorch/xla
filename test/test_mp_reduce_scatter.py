import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  scale = 1 / world_size
  scatter_dim = 1
  shard_size = 2
  input_list_size = 5

  if xm.xla_device_hw(device) in ['TPU', 'CUDA']:
    rand = torch.rand((32, shard_size * world_size, 32))
    xrand = rand.to(device)

    res = xm.reduce_scatter(xm.REDUCE_SUM, xrand, scale, scatter_dim,
                            world_size)
    expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand, scale)
    torch_xla.sync()

    slice_idx = torch.tensor(
        list(range(index * shard_size, (index + 1) * shard_size)))
    expected = expected_world.cpu().index_select(scatter_dim, slice_idx)

    assert res.cpu().allclose(expected)
    xm.rendezvous('test_reduce_scatter')

    # Testing reduce-scatter with list input
    rand_list = [
        torch.rand((32, shard_size * world_size, 32))
        for _ in range(input_list_size)
    ]
    xrand_list = [rand.to(device) for rand in rand_list]

    # TODO: fix the broken case with pin_layout=True
    res_list = xm.reduce_scatter(
        xm.REDUCE_SUM,
        xrand_list,
        scale,
        scatter_dim,
        world_size,
        pin_layout=False)

    for i, res in enumerate(res_list):
      expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand_list[i], scale)
      torch_xla.sync()

      slice_idx = torch.tensor(
          list(range(index * shard_size, (index + 1) * shard_size)))
      expected = expected_world.cpu().index_select(scatter_dim, slice_idx)
      assert res.cpu().allclose(expected)

    xm.rendezvous('test_reduce_scatter_list_input')

    # Testing reduce-scatter with list input bucketized
    rand_list = [
        torch.rand((32, shard_size * world_size, 32))
        for _ in range(input_list_size)
    ]
    xrand_list = [rand.to(device) for rand in rand_list]

    # TODO: fix the broken case with pin_layout=True
    res_list = xm.reduce_scatter_bucketized(
        xm.REDUCE_SUM,
        xrand_list,
        scale,
        scatter_dim,
        world_size,
        pin_layout=False)

    for i, res in enumerate(res_list):
      expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand_list[i], scale)
      torch_xla.sync()

      slice_idx = torch.tensor(
          list(range(index * shard_size, (index + 1) * shard_size)))
      expected = expected_world.cpu().index_select(scatter_dim, slice_idx)
      assert res.cpu().allclose(expected)

    xm.rendezvous('test_reduce_scatter_list_input_bucketized')

    # Testing reduce-scatter with list input and output
    output_list = [
        torch.rand((32, shard_size * world_size, 32))
        for _ in range(input_list_size)
    ]
    xoutput_list = [output.to(device) for output in output_list]

    # TODO: fix the broken case with pin_layout=True
    res_list = xm.reduce_scatter(
        xm.REDUCE_SUM,
        xrand_list,
        scale,
        scatter_dim,
        world_size,
        output=xoutput_list,
        pin_layout=False)

    assert (xoutput_list == res_list)
    for i, res in enumerate(xoutput_list):
      expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand_list[i], scale)
      torch_xla.sync()

      slice_idx = torch.tensor(
          list(range(index * shard_size, (index + 1) * shard_size)))
      expected = expected_world.cpu().index_select(scatter_dim, slice_idx)
      assert res.cpu().allclose(expected)

    xm.rendezvous('test_reduce_scatter_list_input_output')

    # Testing reduce-scatter with list input and output (buckettized)
    output_list = [
        torch.rand((32, shard_size * world_size, 32))
        for _ in range(input_list_size)
    ]
    xoutput_list = [output.to(device) for output in output_list]

    # TODO: fix the broken case with pin_layout=True
    res_list = xm.reduce_scatter_bucketized(
        xm.REDUCE_SUM,
        xrand_list,
        scale,
        scatter_dim,
        world_size,
        output=xoutput_list,
        pin_layout=False)

    assert (xoutput_list == res_list)
    for i, res in enumerate(xoutput_list):
      expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand_list[i], scale)
      torch_xla.sync()

      slice_idx = torch.tensor(
          list(range(index * shard_size, (index + 1) * shard_size)))
      expected = expected_world.cpu().index_select(scatter_dim, slice_idx)
      assert res.cpu().allclose(expected)

    xm.rendezvous('test_reduce_scatter_list_input_output_bucketized')

    # Testing reduce-scatter with list input and output (buckettized, but zero bucket size)
    output_list = [
        torch.rand((32, shard_size * world_size, 32))
        for _ in range(input_list_size)
    ]
    xoutput_list = [output.to(device) for output in output_list]

    # TODO: fix the broken case with pin_layout=True
    res_list = xm.reduce_scatter_bucketized(
        xm.REDUCE_SUM,
        xrand_list,
        scale,
        scatter_dim,
        world_size,
        output=xoutput_list,
        bucket_cap_mb=0,
        pin_layout=False)

    assert (xoutput_list == res_list)
    for i, res in enumerate(xoutput_list):
      expected_world = xm.all_reduce(xm.REDUCE_SUM, xrand_list[i], scale)
      torch_xla.sync()

      slice_idx = torch.tensor(
          list(range(index * shard_size, (index + 1) * shard_size)))
      expected = expected_world.cpu().index_select(scatter_dim, slice_idx)
      assert res.cpu().allclose(expected)

    xm.rendezvous(
        'test_reduce_scatter_list_input_output_bucketized, zero bucket size')
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
