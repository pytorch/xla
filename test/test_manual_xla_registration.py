import sys
import unittest
import torch
from torch import Tensor
from torch.library import custom_op
import torch_xla


@custom_op(
    "xla::custom_einsum",
    schema="(str function, Tensor input, Tensor weight) -> Tensor",
    mutates_args=())
def custom_einsum(function: str, input: Tensor, weight: Tensor):
  return torch.einsum(function, input, weight)


def is_einsum_lowered(func):
  X = torch.zeros(3, 5, requires_grad=False, device='xla')
  Y = torch.zeros(5, 7, requires_grad=False, device='xla')

  out = func(X, Y)
  ir = torch_xla._XLAC._get_xla_tensors_text([out])
  return 'einsum' in ir


class OperationLowered(unittest.TestCase):

  def test_einsum_lowered(self):
    for f in [torch.einsum, custom_einsum]:
      self.assertTrue(
          is_einsum_lowered(lambda a, b: f('...n,mn->...m', a, b)),
          "Operation not lowered; expected operation to be lowered")

  def test_einsum_not_lowered(self):
    self.assertFalse(
        is_einsum_lowered(lambda a, b: torch.einsum('ab,bc->ab', a, b)),
        "Operation lowered; expected operation to not be lowered")


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
