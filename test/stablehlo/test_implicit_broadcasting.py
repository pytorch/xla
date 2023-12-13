import re
import sys
import unittest

import torch
import torch_xla
from torch_xla.core import xla_model as xm
from typing import Tuple, Type, Callable, Union, List

# The following tests cover the implcit-broadcasting for static and bounded
# dynamic shapes.

device = xm.xla_device()


class ImplicitBroadcasting(unittest.TestCase):

  def test_same_rank_broadcast(self):
    a = torch.randn((10, 1)).to(device=device)
    b = torch.randn((1, 5)).to(device=device)
    c = a * b
    stablehlo_text = xm.get_stablehlo([c])
    self.assertTrue(
        re.search(r'stablehlo.multiply.* : tensor<10x5xf32>', stablehlo_text)
        is not None)

  def test_scalar_broadcast(self):
    a = torch.randn(()).to(device=device)
    b = torch.randn((1, 5)).to(device=device)
    c = a * b
    stablehlo_text = xm.get_stablehlo([c])
    self.assertTrue(
        re.search(r'stablehlo.multiply.* : tensor<1x5xf32>', stablehlo_text)
        is not None)

  def test_different_rank_broadcast(self):
    a = torch.randn((10, 1)).to(device=device)
    b = torch.randn((6, 8, 1, 5)).to(device=device)
    c = a * b
    stablehlo_text = xm.get_stablehlo([c])
    self.assertTrue(
        re.search(r'stablehlo.multiply.* : tensor<6x8x10x5xf32>',
                  stablehlo_text) is not None)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
