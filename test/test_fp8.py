import unittest
import torch
import torch_xla
import torch_xla.core.xla_model as xm


class XlaDataTypeTest(unittest.TestCase):

  def test_datatype_fp8e4m3fn(self):
    t1 = torch.tensor([1.0, 2.0, 3.0], device=xm.xla_device())
    t2 = torch.tensor([2.0, 3.0, 4.0], device=xm.xla_device())
    t1 = t1.to(torch.float8_e4m3fn)
    t2 = t2.to(torch.float8_e4m3fn)

    assert t1.dtype == torch.float8_e4m3fn
    assert t2.dtype == torch.float8_e4m3fn
