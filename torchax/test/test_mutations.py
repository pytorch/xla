import unittest
import torchax
import torch
from torch.testing._internal.common_utils import TestCase


class TestMutations(TestCase):

  def setUp(self):
    self.env = torchax.tensor.Environment()
    self.env.config.debug_print_each_op = True

  def test_add(self):
    with self.env:
      x = torch.tensor([1, 2, 3], device='jax', dtype=torch.int32)
      y = torch.tensor([4, 5, 6], device='jax', dtype=torch.int32)
      x.add_(y)
      torch.testing.assert_close(x.cpu(),
                                 torch.tensor([5, 7, 9], dtype=torch.int32))

  def test_sub(self):
    with self.env:
      x = torch.tensor([1, 2, 3], device='jax', dtype=torch.int32)
      y = torch.tensor([4, 5, 6], device='jax', dtype=torch.int32)
      x.sub_(y)
      torch.testing.assert_close(x.cpu(),
                                 torch.tensor([-3, -3, -3], dtype=torch.int32))

  def test_mul(self):
    with self.env:
      x = torch.tensor([1, 2, 3], device='jax', dtype=torch.int32)
      y = torch.tensor([4, 5, 6], device='jax', dtype=torch.int32)

      x.mul_(y)
      torch.testing.assert_close(x.cpu(),
                                 torch.tensor([4, 10, 18], dtype=torch.int32))

  def test_index_copy(self):
    with self.env:
      x = torch.zeros(5, 3, device='jax')
      t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       device='jax',
                       dtype=torch.float)
      index = torch.tensor([0, 4, 2], device='jax')
      x.index_copy_(0, index, t)
      expected = torch.tensor([[1., 2., 3.], [0., 0., 0.], [7., 8., 9.],
                               [0., 0., 0.], [4., 5., 6.]])
      torch.testing.assert_close(x.cpu(), expected)


if __name__ == '__main__':
  unittest.main()
