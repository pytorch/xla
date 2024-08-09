import os
import re
import unittest

import torch
import torch_xla
from torch.utils import _pytree as pytree


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
  
  def test_fp8_scaled_mm(self):
    def to_float8(x, dtype=torch.float8_e5m2):
      finfo = torch.finfo(dtype)
      # Calculate the scale as dtype max divided by absmax
      scale = finfo.max / x.abs().max().clamp(min=1e-12)
      # scale and clamp the tensor to bring it to
      # the representative range of float8 data type
      # (as default cast is unsaturated)
      x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
      # Return both float8 data and the inverse scale (as float),
      # as both required as inputs to torch._scaled_mm
      return x_scl_sat.to(dtype), scale.float().reciprocal()
    
    device = torch_xla.device()
    x = torch.randn((3, 5)).to(torch.bfloat16)
    w = torch.randn((5, 8)).to(torch.bfloat16)
    x_f8, x_inv_s = to_float8(x)
    w_f8, w_inv_s = to_float8(w)
    # args = (x_f8, w_f8, x_inv_s, w_inv_s)
    # args = pytree.tree_map_only(torch.Tensor,
    #                             lambda x: x.to(device), args)
    x_f8 = x_f8.to(device)
    x_inv_s = x_inv_s.to(device)
    w_f8 = w_f8.to(device)
    w_inv_s = w_inv_s.to(device)
    y = torch.ops.aten._scaled_mm(x_f8, w_f8, out_dtype=torch.bfloat16,
                                  scale_a=x_inv_s , scale_b=w_inv_s)
    print(y)



if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
