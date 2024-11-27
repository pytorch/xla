import os
import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

import torch_xla.distributed.spmd.xla_sharding as xs

device = xm.xla_device()

class TestAutocastXla(unittest.TestCase):
  def test_patchedlinear_autocast(self):
    hidden_size = 10
    intermediate_size = 15
    input_tensor = torch.randn(5, hidden_size, requires_grad=True)  # batch_size=5 as example
    linear = torch.nn.Linear(hidden_size, intermediate_size, bias=True)

    with torch.autocast("xla", enabled=True):
      linear = linear.to(device)
      input_tensor = input_tensor.to(device)
      output = xs.xla_patched_nn_linear_forward(linear, input_tensor)
      sum = (output**output).mean()

    sum.backward()
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([linear.weight.grad])

    self.assertTrue(re.search(r".*dot.*bf16", hlo) is not None)

    self.assertTrue(re.search(r".*dot.*f32", hlo) is None)

    bf16_to_f32 = len(re.findall(r".*convert.*f32.*convert.*bf16", hlo))
    f32_to_bf16 = len(re.findall(r".*convert.*bf16.*convert.*f32", hlo))
    self.assertTrue(bf16_to_f32 > 0 and f32_to_bf16 > 0)
    self.assertTrue(bf16_to_f32 > f32_to_bf16)  # cast for grad in backward, so bf16_to_f32 should be higher


if __name__ == "__main__":
  unittest.main()
