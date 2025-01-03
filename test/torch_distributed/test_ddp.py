from absl.testing import absltest, parameterized
import os
import sys
import torch_xla
import torch_xla.core.xla_model as xm

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import args_parse
import distributed_util as util

FLAGS = args_parse.parse_common_options()


class TestXrtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_correctness(rank,
                       use_large_net: bool,
                       debug: bool,
                       gradient_as_bucket_view: bool = False):
    # We cannot run this guard before XMP,
    # see API_GUIDE.md#running-on-multiple-xla-devices-with-multi-processing.
    device = xm.xla_device()
    if xm.xla_device_hw(device) not in ('TPU', 'CUDA'):
      print(
          'Default device {} is not a TPU device'.format(device),
          file=sys.stderr)
      return
    util.ddp_correctness(
        init_method="xla://",
        use_large_net=use_large_net,
        debug=debug,
        gradient_as_bucket_view=gradient_as_bucket_view)

  def test_ddp_correctness(self):
    torch_xla.launch(self._ddp_correctness, args=(False, FLAGS.debug))

  def test_ddp_correctness_with_gradient_as_bucket_view(self):
    torch_xla.launch(self._ddp_correctness, args=(False, FLAGS.debug, True))

  def test_ddp_correctness_large_net(self):
    torch_xla.launch(self._ddp_correctness, args=(True, FLAGS.debug))


if __name__ == "__main__":
  absltest.main()
