import unittest

import torch
import torch_xla2
from torch_xla2 import tensor


class TestContext(unittest.TestCase):
  def test_mode_context_manager(self):
    with torch_xla2.mode():
      x = torch.full((3, 3), -1)
      self.assertIsInstance(x, tensor.XLATensor2)
      y = x.abs()
      self.assertIsInstance(y, tensor.XLATensor2)
      # TODO: remove print
      print(y)

  @staticmethod
  @torch_xla2.mode()
  def _test_mode_decorator():
    x = torch.full((3, 3), -1)
    y = x.abs()

    return x, y

  def test_mode_decorator(self):
    x, y = self._test_mode_decorator()
    self.assertIsInstance(x, tensor.XLATensor2)
    self.assertIsInstance(y, tensor.XLATensor2)
    # TODO: remove print
    print(x, y)


if __name__ == "__main__":
  unittest.main()
