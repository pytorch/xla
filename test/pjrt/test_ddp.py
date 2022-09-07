from absl.testing import absltest, parameterized
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla.experimental import pjrt


def _init_xla_backend(init_file: str):
  rank = xm.get_ordinal()
  world_size = xm.xrt_world_size()

  dist.init_process_group(
      "xla",
      init_method=f"file://{init_file}",
      rank=rank,
      world_size=world_size)


class TestPjRtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_init(init_file: str):
    _init_xla_backend(init_file)

    device = xm.xla_device()
    model = nn.Linear(10, 10).to(device)
    ddp_model = DDP(model)

  def test_ddp_init(self):
    pjrt.run_multiprocess(self._ddp_init, self.create_tempfile().full_path)


if __name__ == "__main__":
  absltest.main()
