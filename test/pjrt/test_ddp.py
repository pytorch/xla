from unittest import mock
from absl.testing import absltest, parameterized
import os
import sys
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt
from torch_xla.experimental import tpu

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import args_parse
import distributed_util as util

FLAGS = args_parse.parse_common_options()

ddp_parameters = [('torch_xla', pjrt.DistributedDataParallel)]
if pjrt.device_type() == 'TPU' and tpu.version() >= 4:
  ddp_parameters.append(('torch', torch.nn.parallel.DistributedDataParallel))


class TestPjRtDistributedDataParallel(parameterized.TestCase):

  def setUp(self):
    os.environ['PJRT_INIT_TORCH_DISTRIBUTED'] = '1'

  @staticmethod
  def _ddp_init(ddp):
    device = xm.xla_device()
    model = nn.Linear(10, 10).to(device)
    ddp_model = ddp(model)

  @parameterized.named_parameters(*ddp_parameters)
  def test_ddp_init(self, ddp: type):
    pjrt._run_multiprocess(self._ddp_init, ddp)

  def test_ddp_init_no_distributed(self):
    with self.assertRaises(RuntimeError):
      with mock.patch.dict(os.environ, {'PJRT_INIT_TORCH_DISTRIBUTED': '0'}):
        pjrt._run_multiprocess(self._ddp_init, pjrt.DistributedDataParallel)

  @parameterized.named_parameters(*ddp_parameters)
  def test_ddp_correctness(self, ddp: type):
    pjrt._run_multiprocess(util.ddp_correctness, ddp=ddp, debug=FLAGS.debug)

  @parameterized.named_parameters(*ddp_parameters)
  def test_ddp_correctness_large_net(self, ddp: type):
    pjrt._run_multiprocess(
        util.ddp_correctness, ddp=ddp, use_large_net=True, debug=FLAGS.debug)


if __name__ == "__main__":
  absltest.main()
