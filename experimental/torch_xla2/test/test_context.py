import unittest

import torch
import torch_xla2
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

  def test_same_manual_seed(self):
    with xla_env:
      torch.manual_seed(1234)
      x = torch.randn((3, 3))
      self.assertIsInstance(x, tensor.XLATensor2)

      torch.manual_seed(1234)
      y = torch.randn((3, 3))
      self.assertIsInstance(y, tensor.XLATensor2)

      print(x, y)

    torch.testing.assert_close(torch_xla2.tensor.j2t(x._elem), torch_xla2.tensor.j2t(y._elem))

if __name__ == "__main__":
  unittest.main()
