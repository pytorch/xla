import os
import re

import torch
import torch_xla
import unittest


class Fp8Test(unittest.TestCase):

  def test_fp8(self):
    device = torch_xla.device()
    fp8_types = [torch.float8_e5m2]
    for dtype in fp8_types:
      t = torch.rand(2, 2).to(dtype)
      xla_t = t.to(device)
      torch_t = xla_t.cpu()
      self.assertEqual(xla_t.dtype, dtype)
      self.assertEqual(torch_t.dtype, dtype)
      # Need to cast to float32 since allclose doesn't work with fp8.
      self.assertTrue(
          torch.allclose(t.to(torch.float32), torch_t.to(torch.float32)))

  def test_fp8_matmul(self):
    device = torch_xla.device()
    fp8_types = [torch.float8_e5m2]
    for dtype in fp8_types:
      t = torch.rand(3, 2).to(dtype)
      w = torch.rand(2, 5).to(dtype)
      torch_matmul = torch.matmul(t, w)
      xla_t = t.to(device)
      xla_w = w.to(device)
      xla_matmul = torch.matmul(xla_t, xla_w)
      xla_matmul = xla_matmul.cpu()
      # Need to cast to float32 since allclose doesn't work with fp8.
      self.assertTrue(
          torch.allclose(
              xla_matmul.to(torch.float32), torch_matmul.to(torch.float32)))

  def test_fp8_hlo(self):
    device = torch_xla.device()
    x = torch.randn((3, 5)).to(torch.float8_e5m2).to(device)
    w = torch.randn((5, 8)).to(torch.float8_e5m2).to(device)
    output = torch.matmul(x, w)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
    self.assertTrue(re.search(r'f8e5m2.*dot.*f8e5m2.*f8e5m2', hlo) is not None)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
