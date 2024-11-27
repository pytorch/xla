import os
import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

device = xm.xla_device()

class TestAutocastXla(unittest.TestCase):
  def test_einsum(self):
    data = torch.randn(16, 10).to(torch.bfloat16).to(device)
    target = torch.randn(5, 10).to(torch.bfloat16).to(device)
    with torch.autocast("xla"):
      product = torch.einsum("...n,mn->...m", data, target)
      hlo = torch_xla._XLAC._get_xla_tensors_hlo([product])

      self.assertTrue(re.search(r".*dot.*bf16", hlo) is not None)

      self.assertTrue(re.search(r".*dot.*f32", hlo) is None)


if __name__ == "__main__":
  unittest.main()
