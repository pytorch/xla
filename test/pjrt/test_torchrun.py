from absl.testing import absltest
from absl import logging
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu
from torch_xla._internal import gpu


class TestTorchrun(absltest.TestCase):

  def test_all_gather(self):
    dist.init_process_group('xla', init_method='xla://')

    dist_world_size = xu.getenv_as('WORLD_SIZE', int)
    devices_per_thread = xr.addressable_device_count()

    expected_world_size = dist_world_size * devices_per_thread

    rank = torch.tensor([dist.get_rank()],
                        dtype=torch.float32,
                        device=xm.xla_device())
    output = [rank.clone() for _ in range(expected_world_size)]
    dist.all_gather(output, rank)
    result = torch.concat(output)
    xm.mark_step()

    expected = torch.arange(0, expected_world_size, step=1, dtype=torch.float32)
    torch.testing.assert_close(result.cpu(), expected)

  def test_xm_all_reduce(self):
    # The test is inspired by https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
    dist.init_process_group('xla', init_method='xla://')

    dist_world_size = xu.getenv_as('WORLD_SIZE', int)
    devices_per_thread = xr.addressable_device_count()
    expected_world_size = dist_world_size * devices_per_thread
    tensors = [torch.arange(2, dtype=torch.int64) + 1 + 2 * r for r in range(expected_world_size)]
    expected = sum(tensors)

    xla_tensor = torch.arange(2, dtype=torch.int64, device=xm.xla_device()) + 1 + 2 * dist.get_rank()
    res = xm.all_reduce(xm.REDUCE_SUM, xla_tensor)
    xm.mark_step()

    torch.testing.assert_close(res.cpu(), expected)

  def test_all_reduce(self):
    # The test is inspired by https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
    dist.init_process_group('xla', init_method='xla://')

    dist_world_size = xu.getenv_as('WORLD_SIZE', int)
    devices_per_thread = xr.addressable_device_count()
    expected_world_size = dist_world_size * devices_per_thread
    tensors = [torch.arange(2, dtype=torch.int64) + 1 + 2 * r for r in range(expected_world_size)]
    expected = sum(tensors)

    xla_tensor = torch.arange(2, dtype=torch.int64, device=xm.xla_device()) + 1 + 2 * dist.get_rank()
    dist.all_reduce(xla_tensor, op=dist.ReduceOp.SUM)
    xm.mark_step()

    torch.testing.assert_close(xla_tensor.cpu(), expected)


if __name__ == '__main__':
  if not dist.is_torchelastic_launched():
    logging.error('Test must be launched with torchrun!')
    exit(1)

  absltest.main()
