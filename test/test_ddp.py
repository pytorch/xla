from absl.testing import absltest, parameterized
import sys
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import args_parse
import distributed_util as util


class TestXrtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_correctness(rank):
    # We cannot run this guard before XMP,
    # see API_GUIDE.md#running-on-multiple-xla-devices-with-multi-processing.
    device = xm.xla_device()
    if xm.xla_device_hw(device) not in ('TPU', 'GPU'):
      print(
          'Default device {} is not a TPU or GPU device'.format(device),
          file=sys.stderr)
      return
    util.ddp_correctness(None)

  def test_ddp_correctness(self):
    xmp.spawn(self._ddp_correctness, args=())


if __name__ == "__main__":
  absltest.main()
