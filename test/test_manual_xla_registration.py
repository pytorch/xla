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
  return ir


class OperationLowered(unittest.TestCase):

  def test_einsum_lowered(self):
    for f in [torch.einsum, custom_einsum]:
      with self.subTest(f=f):
        ir = is_einsum_lowered(lambda a, b: f('...n,mn->...m', a, b))

        self.assertIn(
            "einsum", ir,
            "Expected einsum to be in ir from it being lowered")
        self.assertNotIn(
            "permute", ir,
            "Expected no permute to be in ir from it being lowered")

  def test_einsum_not_lowered(self):
    # 'ab,bc->ab' won't be lowered becaused it cannot be backpropagated
    ir = is_einsum_lowered(lambda a, b: torch.einsum('ab,bc->ab', a, b))

    self.assertNotIn(
        "einsum", ir,
        "Expected no einsum to be in ir from it not being lowered")
    self.assertIn(
        "permute", ir,
        "Expected permute to be in ir from it not being lowered")


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
