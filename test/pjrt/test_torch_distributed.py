from absl.testing import absltest, parameterized
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
import torch_xla.experimental.pjrt_backend
from torch_xla.experimental import pjrt


class TestTorchDistributedPjrt(parameterized.TestCase):

  @staticmethod
  def _all_gather(index: int):
    dist.init_process_group('xla', init_method='pjrt://')
    t = torch.tensor([index], dtype=torch.int32, device=xm.xla_device())
    output = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(output, t)
    output_tensor = torch.concat(output)
    xm.mark_step()

    torch.testing.assert_close(output_tensor.cpu(), torch.arange(0, dist.get_world_size(), 1, dtype=torch.int32))

  def test_all_gather_spawn(self):
    pjrt.spawn(self._all_gather)

  def test_all_gather_spawn_threads(self):
    p = torch.multiprocessing.Process(target=pjrt.spawn_threads, args=(self._all_gather,))
    p.start()
    p.join()

  @staticmethod
  def _ddp_init(index: int):
    dist.init_process_group('xla', init_method='pjrt://')
    device = xm.xla_device()
    model = nn.Linear(10, 10).to(device)
    ddp_model = DDP(model)

  def test_ddp_init_spawn(self):
    pjrt.spawn(self._ddp_init)

  def test_ddp_init_spawn_threads(self):
    p = torch.multiprocessing.Process(target=pjrt.spawn_threads, args=(self._ddp_init,))
    p.start()
    p.join()

if __name__ == '__main__':
  absltest.main()
