from absl.testing import absltest, parameterized
import os
import sys
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import distributed_util as util


class TestPjRtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_init(init_file: str):
    util.init_xla_backend(init_file)

    device = xm.xla_device()
    model = nn.Linear(10, 10).to(device)
    ddp_model = DDP(model)

  def test_ddp_init(self):
    pjrt._run_multiprocess(self._ddp_init, self.create_tempfile().full_path)

  def test_ddp_correctness(self):
    pjrt._run_multiprocess(util.ddp_correctness,
                           self.create_tempfile().full_path)


if __name__ == "__main__":
  absltest.main()
