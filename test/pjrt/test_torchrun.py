from absl.testing import absltest
from absl import logging
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu


class TestTorchrun(absltest.TestCase):
  # TODO(xw32): for the moment, add a teardown to shutdown the 
  # dist server. Need to find a better place tho.

  def test_all_gather(self):
    print('xw32 test_all_gather. Running dist.init_process_group.')
    dist.init_process_group('xla', init_method='xla://')
    print('xw32 test_all_gather. Finished running dist.init_process_group.')

    dist_world_size = xu.getenv_as('WORLD_SIZE', int) # value is 4.
    devices_per_thread = xr.addressable_device_count()
    # RANK env var is also set, and the values are 0,1,2,3.

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


if __name__ == '__main__':
  if not dist.is_torchelastic_launched():
    logging.error('Test must be launched with torchrun!')
    exit(1)

  absltest.main()
