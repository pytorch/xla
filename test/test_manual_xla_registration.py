import sys
import unittest
import torch
from torch import Tensor
from torch.library import custom_op
import torch_xla
import time

@custom_op("xla::custom_linear_with_einsum", schema="(Tensor input, Tensor weight) -> Tensor", mutates_args=())
def custom_linear_with_einsum(input: Tensor, weight: Tensor):
    return torch.einsum('...n,mn->...m', input, weight)

def test_lowering(func):
  X = torch.zeros(3, 3, requires_grad=False, device='xla')
  Y = torch.zeros(3, 3, requires_grad=False, device='xla')

  out = func(X, Y)
  time.sleep(2)
  ir = torch_xla._XLAC._get_xla_tensors_text([out])
  return 'einsum' in ir

class OperationLowered(unittest.TestCase):
  def test_einsum_registration(self):
    self.assertTrue(test_lowering(lambda a, b: torch.einsum('...n,mn->...m', a, b)),
                    "Operation not lowered as expected")

  def test_einsum_custom_registration(self):
    self.assertTrue(test_lowering(lambda a, b: custom_linear_with_einsum(a, b)),
                    "Operation not lowered as expected")

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
