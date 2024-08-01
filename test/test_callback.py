import threading

from absl.testing import absltest
import torch
import torch_xla
from torch_xla.experimental import callback


class TestExperimentalCallback(absltest.TestCase):

  @staticmethod
  @torch_xla.compile
  def executable():
    a, b = torch.randn((100, 100), device=torch_xla.device()), torch.randn(
        (100, 100), device=torch_xla.device())
    return a @ b

  def test_callback(self):
    event = threading.Event()
    c = self.executable()

    def cb(tensor):
      self.assertIs(c, tensor)
      event.set()

    callback.on_ready_callback(c, cb)


if __name__ == "__main__":
  absltest.main()
