from unittest import mock
from absl.testing import absltest, parameterized
import os
import sys
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from torch_xla import runtime as xr
from torch_xla._internal import pjrt, tpu

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import args_parse
import distributed_util as util

FLAGS = args_parse.parse_common_options()


class TestPjRtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_init(index: int = ...):
    dist.init_process_group('xla', init_method='xla://')
    device = torch_xla.device()
    model = nn.Linear(10, 10).to(device)
    ddp_model = DDP(model)

  def test_ddp_init(self):
    pjrt.run_multiprocess(self._ddp_init)

  @absltest.skipIf(xr.device_type() == 'CUDA',
                   "GPU device is not supported by pjrt.spawn_threads")
  def test_ddp_init_threaded(self):
    pjrt.spawn_threads(self._ddp_init)

  @parameterized.named_parameters(('small_net', False), ('large_net', True))
  def test_ddp_correctness(self, use_large_net: bool):
    pjrt.run_multiprocess(
        util.ddp_correctness,
        init_method='xla://',
        use_large_net=use_large_net,
        debug=FLAGS.debug)

  @absltest.skipIf(xr.device_type() == 'TPU' and tpu.version() < 4,
                   "env:// doesn't support multithreading")
  def test_ddp_correctness_env_init(self):
    with mock.patch.dict(os.environ, {
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355'
    }):
      pjrt.run_multiprocess(
          util.ddp_correctness, use_large_net=False, debug=FLAGS.debug)


if __name__ == "__main__":
  absltest.main()
