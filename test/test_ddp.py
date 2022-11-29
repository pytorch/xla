from absl.testing import absltest, parameterized
import sys
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import args_parse
import distributed_util as util

FLAGS = args_parse.parse_common_options()


class TestXrtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_correctness(rank, use_large_net: bool, debug: bool):
    # We cannot run this guard before XMP,
    # see API_GUIDE.md#running-on-multiple-xla-devices-with-multi-processing.
    device = xm.xla_device()
    # TODO(#4049): Enable this test in the GPU environment.
    if xm.xla_device_hw(device) not in ('TPU'):
      print(
          'Default device {} is not a TPU device'.format(device),
          file=sys.stderr)
      return
    util.ddp_correctness(use_large_net=use_large_net, debug=debug)

  def test_ddp_correctness(self):
    xmp.spawn(self._ddp_correctness, args=(False, FLAGS.debug))

  def test_ddp_correctness_large_net(self):
    xmp.spawn(self._ddp_correctness, args=(True, FLAGS.debug))


if __name__ == "__main__":
  absltest.main()
