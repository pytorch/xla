import torch
import torch_xla
from torch_xla import xm
import unittest
from torch_xla.experimental.topp_mask import topp_mask
import math


class TestTopPMask(unittest.TestCase):

  def setUp(self):
    self.device = torch_xla.device()

  def test_invalid_p(self):
    # Test that an invalid p throws an assertion error
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]],
                          dtype=torch.float32,
                          device=self.device)

    with self.assertRaises(AssertionError):
      topp_mask(logits, -0.1)  # p < 0

    with self.assertRaises(AssertionError):
      topp_mask(logits, 1.1)  # p > 1

  def test_basic(self):
    logits = torch.tensor(
        [[math.log(0.2), math.log(0.3),
          math.log(0.5)], [math.log(0.5),
                           math.log(0.2),
                           math.log(0.3)]],
        dtype=torch.float32,
        device=self.device)
    mask = topp_mask(logits, 0.79)

    expected_mask = torch.tensor([[0, 1, 1], [1, 0, 1]],
                                 dtype=torch.float32,
                                 device=self.device)
    self.assertTrue(torch.allclose(expected_mask, mask, atol=1e-6))

  def test_dim(self):
    logits = torch.tensor([[math.log(0.2), math.log(0.3)],
                           [math.log(0.3), math.log(0.2)],
                           [math.log(0.5), math.log(0.5)]],
                          dtype=torch.float32,
                          device=self.device)
    mask = topp_mask(logits, 0.79, dim=0)

    expected_mask = torch.tensor([[0, 1], [1, 0], [1, 1]],
                                 dtype=torch.float32,
                                 device=self.device)
    self.assertTrue(torch.allclose(expected_mask, mask, atol=1e-6))

  def test_p_is_zero(self):
    logits = torch.tensor([[0.2, 0.5, 5], [0.1, 2, 0.2]],
                          dtype=torch.float32,
                          device=self.device)
    mask = topp_mask(logits, 0.0)

    expected_mask = torch.tensor([[0, 0, 1], [0, 1, 0]],
                                 dtype=torch.float32,
                                 device=self.device)
    self.assertTrue(torch.allclose(expected_mask, mask, atol=1e-6))

  def test_p_is_one(self):
    logits = torch.tensor([[0.2, 0.5, 5], [0.1, 2, 0.2]],
                          dtype=torch.float32,
                          device=self.device)
    mask = topp_mask(logits, 1.0)

    # All elements should be selected.
    expected_mask = torch.tensor([[1, 1, 1], [1, 1, 1]],
                                 dtype=torch.float32,
                                 device=self.device)
    self.assertTrue(torch.allclose(expected_mask, mask, atol=1e-6))


if __name__ == '__main__':
  unittest.main()
