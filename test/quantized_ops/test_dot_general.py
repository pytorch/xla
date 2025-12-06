import os
import re
import torch
import torch_xla
import unittest
device = torch_xla.device()
torch.manual_seed(12345)
class DotGeneralTest(unittest.TestCase):
  def test_dot_general(self):
    x = torch.rand(10, 3, 4).to(torch.bfloat16)
    w = torch.rand(10, 4, 5).to(torch.bfloat16)
    expected_out = torch.matmul(x, w)
    xla_x = x.to(device)
    xla_w = w.to(device)
    xla_out = torch_xla._XLAC._xla_dot_general(xla_x, xla_w,
                                               (([2], [1]), ([0], [0])))
    self.assertTrue(torch.allclose(xla_out.cpu(), expected_out))
  def test_dot_general_negative_dim(self):
    x = torch.rand(10, 3, 4).to(torch.bfloat16)
    w = torch.rand(10, 4, 5).to(torch.bfloat16)
    expected_out = torch.matmul(x, w)
    xla_x = x.to(device)
    xla_w = w.to(device)
    xla_out = torch_xla._XLAC._xla_dot_general(xla_x, xla_w,
                                               (([-1], [-2]), ([0], [0])))
    self.assertTrue(torch.allclose(xla_out.cpu(), expected_out))
  def test_dot_general_linear(self):
    x = torch.rand(10, 3, 4).to(torch.bfloat16)
    w = torch.rand(5, 4).to(torch.bfloat16)
    expected_out = torch.matmul(x, w.t())
    xla_x = x.to(device)
    xla_w = w.to(device)
    xla_out = torch_xla._XLAC._xla_dot_general(xla_x, xla_w, (([2], [1]), ()))
    self.assertTrue(torch.allclose(xla_out.cpu(), expected_out))
  def test_dot_general_int32_dtype(self):
    x = torch.randint(-15, 15, (10, 3, 4)).to(torch.int32)
    w = torch.randint(-15, 15, (10, 4, 5)).to(torch.int32)
    # Though the intention of the test is for: matmul(int8, int8)->int32
    # Do not cast to int8 for eager, since torch only has matmul(int32, int32)->in32
    # Casting to int8 would cause the result overflow.
    expected_out = torch.matmul(x, w)
    xla_x = x.to(torch.int8).to(device)
    xla_w = w.to(torch.int8).to(device)
    xla_out = torch_xla._XLAC._xla_dot_general(
        xla_x,
        xla_w, (([2], [1]), ([0], [0])),
        preferred_element_type=torch.int32)
    self.assertTrue(torch.allclose(xla_out.cpu(), expected_out))
  def test_raises_error_on_non_xla_tensor(self):
    lhs = torch.rand(10, 3, 4, dtype=torch.bfloat16)
    rhs = torch.rand(10, 4, 5, dtype=torch.bfloat16)
    def test(args, non_xla_tensor_arg):
      arg_number_to_str = ["first", "second"]
      position = arg_number_to_str[non_xla_tensor_arg]
      try:
        torch_xla._XLAC._xla_dot_general(*args, (([2], [1]), ([0], [0])))
      except RuntimeError as err:
        error_message = (
            f"Expected input tensor ({position} argument) to be an actual XLA tensor. "
            f"Got: CPUBFloat16Type. Consider moving it ({position} argument) to XLA."
        )
        msg = str(err)
        self.assertIn(error_message, msg)
    test((lhs, rhs.to(device)), non_xla_tensor_arg=0)
    test((lhs.to(device), rhs), non_xla_tensor_arg=1)
if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
