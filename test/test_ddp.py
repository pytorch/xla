from absl.testing import absltest, parameterized
import pjrt.distributed_util as util
import sys
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm


class TestXrtDistributedDataParallel(parameterized.TestCase):

  @staticmethod
  def _ddp_correctness(rank):
    util.ddp_correctness(None)

  def test_ddp_correctness(self):
    device = xm.xla_device()
    if xm.xla_device_hw(device) not in ('TPU', 'GPU'):
      print(
          'Default device {} is not a TPU or GPU device'.format(device),
          file=sys.stderr)
      return

    xmp.spawn(self._ddp_correctness, args=())


if __name__ == "__main__":
  absltest.main()
