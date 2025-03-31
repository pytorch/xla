import os
import re

import torch
import torch_xla
import unittest
from absl.testing import parameterized

device = torch_xla.device()

dtype_parameters = [
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

torch.manual_seed(123456)

class Fp8Test(parameterized.TestCase):

  @parameterized.parameters(*dtype_parameters)
  def test_fp8(self, dtype):
    t = torch.rand(2, 2).to(dtype)
    xla_t = t.to(device)
    torch_t = xla_t.cpu()
    self.assertEqual(xla_t.dtype, dtype)
    self.assertEqual(torch_t.dtype, dtype)
    # Need to cast to float32 since allclose doesn't work with fp8.
    self.assertTrue(
        torch.allclose(t.to(torch.float32), torch_t.to(torch.float32)))

  @parameterized.parameters(*dtype_parameters)
  def test_fp8_matmul(self, dtype):
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

  @parameterized.parameters(*dtype_parameters)
  def test_fp8_hlo(self, dtype):
    x = torch.randn((3, 5)).to(dtype).to(device)
    w = torch.randn((5, 8)).to(dtype).to(device)
    output = torch.matmul(x, w)
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
    exmy_str = str(dtype).split('_')[-1]
    self.assertTrue(
        re.search(rf'f8{exmy_str}.*dot.*f8{exmy_str}.*f8{exmy_str}', hlo)
        is not None)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
