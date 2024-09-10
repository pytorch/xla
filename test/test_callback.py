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
      # TODO: check that result is both assigned and completed
      self.assertNotIn("Data Handle: None",
                       torch_xla._XLAC._get_xla_tensor_debug_info(tensor))
      event.set()

    callback.on_ready_callback(c, cb)
    event.wait(3)

  def test_callback_event(self):
    c = self.executable()
    c_ready_event = callback.on_ready_event(c)
    c_ready_event.wait(3)
    self.assertNotIn("Data Handle: None",
                     torch_xla._XLAC._get_xla_tensor_debug_info(c))


if __name__ == "__main__":
  absltest.main()
