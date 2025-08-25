import sys
import unittest
import test_xla_sharding_base
from torch.distributed.tensor import DTensor
from torch_xla.distributed.spmd import XLAShardedTensor

import torch


class XlaShardedTensorTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_xlashardedtensor_is_dtensor(self):
    """Test that XLAShardedTensor is a subclass of DTensor."""
    xt = torch.randn(128, 128).to('xla')
    xla_tensor = XLAShardedTensor(xt)
    self.assertIsInstance(xla_tensor, DTensor)

  def test_xlashardedtensor_gradient(self):
    """Test accessing gradients of an XLAShardedTensor (triggers __torch_function__)."""
    xt = torch.randn(128, 128).to('xla')
    xla_tensor = XLAShardedTensor(xt, requires_grad=True)
    result = xla_tensor.sum()
    result.backward()

    # this should trigger __torch_function__
    grad = xla_tensor.grad

    self.assertIsNotNone(grad)
    self.assertEqual(grad.shape, xla_tensor.shape)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
