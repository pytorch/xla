import torch
import torchax
import re
import sys
import unittest

from torchax.tensor import Tensor
from torchax.view import View


class TrainTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    torchax.enable_globally()

  def test_copy_(self):
    x = torch.zeros((10, 10), device="jax")
    y = torch.ones((5, 5), device="jax")
    x[0:5, :][:, 0:5].copy_(y[:, :])
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x[0:5, 0:5].sum(), 25)
    self.assertEqual(x.sum(), 25)

  def test_transivity(self):
    x = torch.zeros((10, 10), device="jax")
    x_view = x[0:5, :][:, 0:5].add_(1)
    y_view = x_view[0:5, :][:, 0:5].add_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(type(x_view), View)
    self.assertEqual(type(y_view), View)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x[0:5, 0:5].sum(), 50)
    self.assertEqual(x.sum(), 50)

  def test_outofplace_add(self):
    x = torch.zeros((10, 10), device="jax")
    x2 = x[0:5, :][:, 0:5].add(1)
    x3 = x2[0:5, :][:, 0:5].add_(x2)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(type(x2), Tensor)
    self.assertEqual(type(x3), View)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 0)
    self.assertEqual(x2.sum(), 50)

  def test_multiply_tensor_and_view(self):
    x = torch.ones((10, 10), device="jax") * 2
    y = torch.ones((10, 10), device="jax")
    x1 = x[:, :]
    res = x1.mul(y)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(type(y), Tensor)
    self.assertEqual(type(x1), View)
    self.assertEqual(type(res), Tensor)
    self.assertEqual(res.sum(), 200)

  def test_multiply_views(self):
    x = torch.ones((10, 10), device="jax") * 2
    y = torch.ones((10, 10), device="jax")
    x1 = x[0:1, :]
    y1 = y[0:1, :]
    res = x1.mul(y1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(type(y), Tensor)
    self.assertEqual(type(x1), View)
    self.assertEqual(type(y1), View)
    self.assertEqual(type(res), Tensor)
    self.assertEqual(res.sum(), 20)

  def test_setitem(self):
    a = torch.zeros(10, device="jax")
    a[0:5][0:3] = 1
    self.assertEqual(type(a), Tensor)
    self.assertEqual(a.shape, (10,))
    self.assertEqual(a.sum(), 3)

  # Test all in-place operations
  def test_add_(self):
    x = torch.zeros((10, 10), device="jax")
    x[0:5, :][:, 0:5].add_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 25)

  def test_sub_(self):
    x = torch.zeros((10, 10), device="jax")
    x[0:5, :][:, 0:5].sub_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), -25)

  def test_mul_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:5, :][:, 0:5].mul_(2)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 125)

  def test_div_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:10, :][:, 0:10].div_(2)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 50)

  def test_pow_(self):
    x = torch.full((10, 10), fill_value=2, device="jax")
    x[0:5, :][:, 0:5].pow_(2)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 250)

  def test_clamp_(self):
    x = torch.arange(100, device="jax", dtype=torch.float).reshape(10, 10)
    x[0:5, :][:, 0:5].clamp_(min=50, max=80)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertTrue((x[0:5, 0:5] >= 50).all())
    self.assertTrue((x[0:5, 0:5] <= 80).all())

  def test_lt_(self):
    x = torch.ones((10, 10), device="jax")
    y = torch.zeros((10, 10), device="jax")
    x[0:5, :][:, 0:5].lt_(0.5)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x[0:5, 0:5].sum(),
                     0)  # All False (0) in the modified region
    self.assertEqual(x[5:, 5:].sum(),
                     25)  # All True (1) in the unmodified region

  def test_le_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:5, :][:, 0:5].le_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 100)  # All True (1)

  def test_gt_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:5, :][:, 0:5].gt_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x[0:5, 0:5].sum(),
                     0)  # All False (0) in the modified region
    self.assertEqual(x.sum(), 75)  # Only the unmodified region is True (1)

  def test_ge_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:5, :][:, 0:5].ge_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 100)  # All True (1)

  def test_eq_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:5, :][:, 0:5].eq_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x.sum(), 100)  # All True (1)

  def test_ne_(self):
    x = torch.ones((10, 10), device="jax")
    x[0:5, :][:, 0:5].ne_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    self.assertEqual(x[0:5, 0:5].sum(),
                     0)  # All False (0) in the modified region
    self.assertEqual(x.sum(), 75)  # Only the unmodified region is True (1)

  def test_bernoulli_(self):
    # Set a fixed seed for deterministic behavior
    torch.manual_seed(42)
    x = torch.full((10, 10), fill_value=0.5, device="jax")
    y = x[0:5, :][:, 0:5]
    y.bernoulli_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Values will be 0 or 1 in the modified region
    self.assertTrue(((x[0:5, 0:5] == 0) | (x[0:5, 0:5] == 1)).all())
    # Unmodified region remains 0.5
    self.assertTrue((x[5:, 5:] == 0.5).all())

  def test_geometric_(self):
    torch.manual_seed(42)
    x = torch.full((10, 10), fill_value=0.5, device="jax")
    y = x[0:5, :][:, 0:5]
    y.geometric_(p=0.5)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Geometric distribution values are positive integers
    self.assertTrue((x[0:5, 0:5] >= 1).all())
    # Unmodified region remains 0.5
    self.assertTrue((x[5:, 5:] == 0.5).all())

  def test_normal_(self):
    torch.manual_seed(42)
    x = torch.zeros((10, 10), device="jax")
    x[0:5, :][:, 0:5].normal_(mean=0, std=1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Unmodified region remains 0
    self.assertEqual(x[5:, 5:].sum(), 0)

  def test_uniform_(self):
    torch.manual_seed(42)
    x = torch.zeros((10, 10), device="jax")
    x[0:5, :][:, 0:5].uniform_(0, 1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Values in modified region are between 0 and 1
    self.assertTrue((x[0:5, 0:5] >= 0).all())
    self.assertTrue((x[0:5, 0:5] <= 1).all())
    # Unmodified region remains 0
    self.assertEqual(x[5:, 5:].sum(), 0)

  def test_relu_(self):
    x = torch.randn((10, 10), device="jax")
    x_copy = x.clone()
    x[0:5, :][:, 0:5].relu_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Modified region has no negative values
    self.assertTrue((x[0:5, 0:5] >= 0).all())
    # Unmodified region remains the same
    self.assertTrue(torch.equal(x[5:, 5:], x_copy[5:, 5:]))

  def test_squeeze_(self):
    x = torch.randn((10, 1, 10), device="jax")
    x_clone = x.clone()
    # Squeeze the middle dimension
    x.squeeze_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Content should remain the same
    self.assertTrue(torch.allclose(x, x_clone.squeeze()))

  def test_sqrt_(self):
    x = torch.randn((10, 10),
                    device="jax").abs()  # Use abs to ensure positive values
    x_copy = x.clone()
    x[0:5, :][:, 0:5].sqrt_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Modified region is square root of original values
    self.assertTrue(torch.allclose(x[0:5, 0:5], torch.sqrt(x_copy[0:5, 0:5])))
    # Unmodified region remains the same
    self.assertTrue(torch.equal(x[5:, 5:], x_copy[5:, 5:]))

  def test_clamp_min_(self):
    x = torch.randn((10, 10), device="jax")
    x_copy = x.clone()
    x[0:5, :][:, 0:5].clamp_min_(0)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Modified region has no values below 0
    self.assertTrue((x[0:5, 0:5] >= 0).all())
    # Unmodified region remains the same
    self.assertTrue(torch.equal(x[5:, 5:], x_copy[5:, 5:]))

  def test_sigmoid_(self):
    x = torch.randn((10, 10), device="jax")
    x_copy = x.clone()
    x[0:5, :][:, 0:5].sigmoid_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Modified region values are between 0 and 1
    self.assertTrue((x[0:5, 0:5] >= 0).all())
    self.assertTrue((x[0:5, 0:5] <= 1).all())
    # Unmodified region remains the same
    self.assertTrue(torch.equal(x[5:, 5:], x_copy[5:, 5:]))

  def test_tanh_(self):
    x = torch.randn((10, 10), device="jax")
    x_copy = x.clone()
    x[0:5, :][:, 0:5].tanh_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Modified region values are between -1 and 1
    self.assertTrue((x[0:5, 0:5] >= -1).all())
    self.assertTrue((x[0:5, 0:5] <= 1).all())
    # Unmodified region remains the same
    self.assertTrue(torch.equal(x[5:, 5:], x_copy[5:, 5:]))

  def test_ceil_(self):
    x = torch.randn((10, 10), device="jax")
    x_copy = x.clone()
    x[0:5, :][:, 0:5].ceil_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Check that ceil operation was applied correctly
    self.assertTrue(torch.allclose(x[0:5, 0:5], torch.ceil(x_copy[0:5, 0:5])))
    # Unmodified region remains the same
    self.assertTrue(torch.equal(x[5:, 5:], x_copy[5:, 5:]))

  def test_logical_not_(self):
    x = torch.zeros((10, 10), device="jax")
    x[0:5, 0:5] = 1  # Set some values to 1
    x[0:5, :][:, 0:5].logical_not_()
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Modified region has all values flipped
    self.assertEqual(x[0:5, 0:5].sum(), 0)  # All now 0
    # Unmodified region remains 0
    self.assertEqual(x[5:, 5:].sum(), 0)

  def test_unsqueeze_(self):
    x = torch.randn((10, 10), device="jax")
    x_copy = x.clone()
    # Add dimension at index 1
    x.unsqueeze_(1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 1, 10))
    # Content should remain the same
    self.assertTrue(torch.equal(x.squeeze(1), x_copy))

  def test_transpose_(self):
    x = torch.randn((10, 5), device="jax")
    x_copy = x.clone()
    # Transpose dimensions 0 and 1
    x.transpose_(0, 1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (5, 10))
    # Check transposition worked correctly
    self.assertTrue(torch.equal(x, x_copy.transpose(0, 1)))

  def test_log_normal_(self):
    torch.manual_seed(42)
    x = torch.zeros((10, 10), device="jax")
    x[0:5, :][:, 0:5].log_normal_(mean=0, std=1)
    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (10, 10))
    # Log-normal values are positive
    self.assertTrue((x[0:5, 0:5] > 0).all())
    # Unmodified region remains 0
    self.assertEqual(x[5:, 5:].sum(), 0)

  def test_scatter_add_(self):
    # Initialize test tensors
    x = torch.zeros((5, 5), device="jax")
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device="jax")
    values = torch.ones((2, 3), device="jax")

    # Apply scatter_add_ operation
    x.scatter_add_(0, indices, values)

    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (5, 5))
    # Check specific values were added
    self.assertTrue(torch.all(x[0, 0] == 2.0))
    self.assertEqual(x.sum(), 6.0)  # Only the 3 specified positions have values

  def test_scatter_(self):
    # Initialize test tensors
    x = torch.zeros((5, 5), device="jax")
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device="jax")
    values = torch.ones((2, 3), device="jax") * 2.0

    # Apply scatter_ operation
    x.scatter_(0, indices, values)

    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (5, 5))
    # Check specific values were replaced
    self.assertEqual(x[0, 0], 2.0)
    self.assertEqual(x[1, 1], 2.0)
    self.assertEqual(x[2, 2], 2.0)
    self.assertEqual(x.sum(), 6.0)  # Only the 3 specified positions have values

  def test_scatter_reduce_(self):
    # Initialize test tensors
    x = torch.ones((5, 5), device="jax")
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device="jax")
    values = torch.ones((2, 3), device="jax") * 2.0

    # Apply scatter_reduce_ operation with "sum" reduction
    x.scatter_reduce_(0, indices, values, reduce="sum")

    self.assertEqual(type(x), Tensor)
    self.assertEqual(x.shape, (5, 5))
    # Check specific values were reduced
    self.assertTrue(torch.all(x[0, 0] == 5.0))
    self.assertEqual(x.sum(), 37.0)
