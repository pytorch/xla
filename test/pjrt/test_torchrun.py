from absl.testing import absltest
from absl import logging
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu


class TestTorchrun(absltest.TestCase):

  def setUp(self):
    dist.init_process_group('xla', init_method='xla://')

  def tearDown(self) -> None:
    dist.destroy_process_group()

  def test_addressable_device_count(self):
    devices_per_process = xr.addressable_device_count()
    self.assertEqual(devices_per_process, 1)

  def test_all_gather(self):
    dist_world_size = xu.getenv_as('WORLD_SIZE', int)
    devices_per_thread = xr.addressable_device_count()

    expected_world_size = dist_world_size * devices_per_thread

    rank = torch.tensor([dist.get_rank()],
                        dtype=torch.float32,
                        device=xm.xla_device())
    output = [rank.clone() for _ in range(expected_world_size)]
    dist.all_gather(output, rank)
    result = torch.concat(output)
    torch_xla.sync()

    expected = torch.arange(0, expected_world_size, step=1, dtype=torch.float32)
    torch.testing.assert_close(result.cpu(), expected)

  def test_all_reduce(self):
    # The test is inspired by https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
    dist_world_size = xu.getenv_as('WORLD_SIZE', int)
    devices_per_thread = xr.addressable_device_count()
    world_size = dist_world_size * devices_per_thread

    # If world_size=2, then the `tensors` below will be [[1, 2], [3, 4]].
    # The `expected` will be [4, 6].
    tensors = [
        torch.arange(2, dtype=torch.int64) + 1 + 2 * r
        for r in range(world_size)
    ]
    expected = sum(tensors)

    xla_tensor = torch.arange(
        2, dtype=torch.int64, device=xm.xla_device()) + 1 + 2 * dist.get_rank()
    dist.all_reduce(xla_tensor, op=dist.ReduceOp.SUM)
    torch_xla.sync()

    torch.testing.assert_close(xla_tensor.cpu(), expected)

  def test_reduce_scatter(self):
    # The test is inspired by https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter
    dist_world_size = xu.getenv_as('WORLD_SIZE', int)
    devices_per_thread = xr.addressable_device_count()
    world_size = dist_world_size * devices_per_thread
    # If world_size=2, then `tensor` will be tensor([0, 2, 4, 6])
    # `expected` will be [0, 2] on rank 0 and [4, 6] on rank 1.
    tensor = world_size * torch.arange(
        world_size * world_size, dtype=torch.int64)
    expected = torch.split(tensor, world_size)[dist.get_rank()]

    tensor_out = torch.zeros(
        world_size, dtype=torch.int64, device=xm.xla_device())
    tensor_in = torch.arange(
        world_size * world_size, dtype=torch.int64, device=xm.xla_device())
    dist.reduce_scatter(tensor_out, [tensor_in], op=dist.ReduceOp.SUM)
    torch_xla.sync()

    torch.testing.assert_close(tensor_out.cpu(), expected)


if __name__ == '__main__':
  if not dist.is_torchelastic_launched():
    logging.error('Test must be launched with torchrun!')
    exit(1)

  absltest.main()
