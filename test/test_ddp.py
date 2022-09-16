from absl.testing import absltest, parameterized
import torch_xla.distributed.xla_multiprocessing as xmp
import pjrt.distributed_util as util


class TestXrtDistributedDataParallel(parameterized.TestCase):
  @staticmethod
  def _ddp_correctness(rank):
    util.ddp_correctness(None)

  def test_ddp_correctness(self):
    xmp.spawn(self._ddp_correctness, args=())


if __name__ == "__main__":
  absltest.main()
