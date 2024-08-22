import sys

import torch
import torch_xla
import unittest
import torch_xla.core.xla_model as xm

device = xm.xla_device()


class Fp8Test(unittest.TestCase):

  def test_fp8(self):
    dtype = torch.float8_e4m3fn
    t = torch.rand(2, 2).to(dtype)
    xla_t = t.to(device)
    torch_t = xla_t.cpu()
    self.assertEqual(xla_t.dtype, dtype)
    self.assertEqual(torch_t.dtype, dtype)
    # Need to cast to float32 since allclose doesn't work with fp8.
    self.assertTrue(
        torch.allclose(t.to(torch.float32), torch_t.to(torch.float32)))

  def test_fp8_matmul(self):
    dtype = torch.float8_e4m3fn
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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
