import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

device = xm.xla_device()


class TestAutocastXla(unittest.TestCase):
  def test_cross_entropy_loss(self):
    data = torch.randn(16, 10).to(torch.bfloat16).to(device)
    target = torch.randn(16, 10).to(torch.bfloat16).to(device)

    with torch.autocast("xla"):
      loss = torch.nn.CrossEntropyLoss()(data, target)
      hlo = torch_xla._XLAC._get_xla_tensors_hlo([loss])
      self.assertRegex(hlo, r".*convert.*f32.*convert.*bf16")
      self.assertRegex(hlo, r".*exponential.*f32.*exponential.*f32")
      self.assertRegex(hlo, r".*log.*f32.*log.*f32")

  def test_einsum(self):
    # irrespective of input dtype, output dtype will depend on autocast policy.
    # Tests for bf16 and f32 given below.

    # input data of type bf16
    data = torch.randn(16, 10).to(torch.bfloat16).to(device)
    target = torch.randn(5, 10).to(torch.bfloat16).to(device)

    with torch.autocast("xla"):
      product = torch.einsum("...n,mn->...m", data, target)
      # test the HLO to see if autocast works for einsum op, which would show up as a dot op in the HLO
      hlo = torch_xla._XLAC._get_xla_tensors_hlo([product])
      # Verify that dot op has bf16 output and not f32, i.e. the computation is performed in bfloat16 precision by autocast
      self.assertRegex(hlo, r".*dot.*bf16")
      self.assertNotRegex(hlo, r".*dot.*f32")

    # input data of type fp32
    data32 = torch.randn(16, 10).to(torch.float32).to(device)
    target32 = torch.randn(5, 10).to(torch.float32).to(device)

    with torch.autocast("xla"):
      product = torch.einsum("...n,mn->...m", data32, target32)
      # test the HLO to see if autocast works for einsum op, which would show up as a dot op in the HLO
      hlo = torch_xla._XLAC._get_xla_tensors_hlo([product])
      # Verify that dot op has bf16 output and not f32, i.e. the computation is performed in bfloat16 precision by autocast
      self.assertRegex(hlo, r".*dot.*bf16")
      self.assertNotRegex(hlo, r".*dot.*f32")


if __name__ == "__main__":
  unittest.main()
