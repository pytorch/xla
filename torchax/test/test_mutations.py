import unittest
import torchax
import torch
from torch.testing._internal.common_utils import TestCase


class TestMutations(TestCase):

  def setUp(self):
    self.env = torchax.tensor.Environment()

  def test_add(self):
    with self.env:
      x = torch.tensor([1, 2, 3], dtype=torch.int32)
      y = torch.tensor([4, 5, 6], dtype=torch.int32)
      x.add_(y)
      self.assertEqual(x, torch.tensor([5, 7, 9], dtype=torch.int32))

  def test_sub(self):
    with self.env:
      x = torch.tensor([1, 2, 3], dtype=torch.int32)
      y = torch.tensor([4, 5, 6], dtype=torch.int32)
      x.sub_(y)
      self.assertEqual(x, torch.tensor([-3, -3, -3], dtype=torch.int32))

  def test_mul(self):
    with self.env:
      x = torch.tensor([1, 2, 3], dtype=torch.int32)
      y = torch.tensor([4, 5, 6], dtype=torch.int32)

      x.mul_(y)
      self.assertEqual(x, torch.tensor([4, 10, 18], dtype=torch.int32))


if __name__ == '__main__':
  unittest.main()
