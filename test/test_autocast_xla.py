import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

import torch_xla.distributed.spmd.xla_sharding as xs

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

  def test_patchedlinear_autocast(self):
    hidden_size = 10
    intermediate_size = 15
    input_tensor = torch.randn(
        5, hidden_size, requires_grad=True)  # batch_size=5 as example
    linear = torch.nn.Linear(hidden_size, intermediate_size, bias=True)

    with torch.autocast("xla", enabled=True):
      linear = linear.to(device)
      input_tensor = input_tensor.to(device)
      output = xs.xla_patched_nn_linear_forward(linear, input_tensor)
      result = (output**output).mean()
      # a simple sum would suffice, but then hlo for linear.weight.grad would be only the backward pass
      #  since grad of sum is constant. Hence this is done, to get whole execution graph in HLO.

    result.backward()
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([linear.weight.grad])
    # Verify that matrix multiplication is performed in bfloat16 precision and not f32.
    # XLAPatchedLinear uses einsum instead of matmul, hence this is checking if einsum op in this layer is being autocast.
    self.assertRegex(hlo, r".*dot.*bf16")
    self.assertNotRegex(hlo, r".*dot.*f32")

    bf16_to_f32 = len(re.findall(r".*convert.*f32.*convert.*bf16", hlo))
    f32_to_bf16 = len(re.findall(r".*convert.*bf16.*convert.*f32", hlo))

    # Verify that precision conversions are happening
    self.assertTrue(bf16_to_f32 > 0 and f32_to_bf16 > 0)
    # Verify more bf16->f32 conversions than f32->bf16, since this is expected during backward pass for grad computation
    self.assertTrue(bf16_to_f32 == 11)
    self.assertTrue(f32_to_bf16 == 8)
    self.assertTrue(
        bf16_to_f32 > f32_to_bf16
    )  #redundant given the above two, but this is what we actually want to verify

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
