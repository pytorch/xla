from absl.testing import absltest, parameterized
from absl import logging
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla.experimental import pjrt
from .. import distributed_util as util


class TestPjRtDistributedDataParallel(parameterized.TestCase):

  def test_ddp_init(self):
    pjrt.run_multiprocess(util.ddp_init, self.create_tempfile().full_path)

  def test_ddp_correctness(self):
    pjrt.run_multiprocess(self.ddp_correctness,
                          self.create_tempfile().full_path)


if __name__ == "__main__":
  absltest.main()
