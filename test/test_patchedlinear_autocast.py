import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

import torch_xla.distributed.spmd.xla_sharding as xs

device = xm.xla_device()

class TestPatchedLinearAutocastXla(unittest.TestCase):
  def test_patchedlinear_autocast(self):
    hidden_size = 10
    intermediate_size = 15
    input_tensor = torch.randn(5, hidden_size, requires_grad=True)  # batch_size=5 as example
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
    # XLAPatchedLinear uses einsum instead of matmul, hence this is basically checking if einsum op in this layer is being autocast.
    self.assertTrue(re.search(r".*dot.*bf16", hlo) is not None)
    self.assertTrue(re.search(r".*dot.*f32", hlo) is None)

    bf16_to_f32 = len(re.findall(r".*convert.*f32.*convert.*bf16", hlo))
    f32_to_bf16 = len(re.findall(r".*convert.*bf16.*convert.*f32", hlo))
    # Verify that precision conversions are happening
    self.assertTrue(bf16_to_f32 > 0 and f32_to_bf16 > 0)
    # Verify more bf16->f32 conversions than f32->bf16, since this is expected during backward pass for grad computation
    self.assertTrue(bf16_to_f32 > f32_to_bf16)


if __name__ == "__main__":
  unittest.main()
