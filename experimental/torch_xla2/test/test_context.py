import unittest

import torch
from torch_xla2 import tensor

xla_env = tensor.Environment(0)


class TestContext(unittest.TestCase):

  def test_mode_context_manager(self):
    with xla_env:
      x = torch.full((3, 3), -1)
      self.assertIsInstance(x, tensor.XLATensor2)
      y = x.abs()
      self.assertIsInstance(y, tensor.XLATensor2)

  @staticmethod
  @xla_env
  def _test_mode_decorator():
    x = torch.full((3, 3), -1)
    y = x.abs()

    return x, y

  def test_mode_decorator(self):
    x, y = self._test_mode_decorator()
    self.assertIsInstance(x, tensor.XLATensor2)
    self.assertIsInstance(y, tensor.XLATensor2)


if __name__ == "__main__":
  unittest.main()
