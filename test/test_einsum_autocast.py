import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

device = xm.xla_device()

class TestEinsumAutocastXla(unittest.TestCase):
  def test_einsum(self):
    data = torch.randn(16, 10).to(torch.bfloat16).to(device)
    target = torch.randn(5, 10).to(torch.bfloat16).to(device)

    with torch.autocast("xla"):
      product = torch.einsum("...n,mn->...m", data, target)
      # test the HLO to see if autocast works for einsum op, which would show up as a dot op in the HLO
      hlo = torch_xla._XLAC._get_xla_tensors_hlo([product])
      # Verify that dot op has bf16 output and not f32, i.e. the computation is performed in bfloat16 precision by autocast
      self.assertTrue(re.search(r".*dot.*bf16", hlo) is not None)
      self.assertTrue(re.search(r".*dot.*f32", hlo) is None)


if __name__ == "__main__":
  unittest.main()
