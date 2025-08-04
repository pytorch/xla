import sys
import unittest
import torch
import numpy as np

from torch.distributed.tensor import DeviceMesh
from torch.distributed._tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd.xla_sharded_tensor import XLAShardedTensor
import test_xla_sharding_base


class DTensorXLAFromLocalConversionTest(test_xla_sharding_base.XlaShardingTest):
  """
    Test suite for the automatic conversion of regular tensors to XLAShardedTensor
    in DTensor.from_local() when using XLA device mesh.
    """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_to_local(self):
    from torch.distributed.tensor import distribute_tensor
    world_size = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", list(range(world_size)))

    big_tensor = torch.randn(100000, 88)
    sharded_tensor = XLAShardedTensor(big_tensor, mesh, [Shard(0)])

    local_tensor = sharded_tensor.to_local()

    # Verify the shapes are the same
    self.assertEqual(local_tensor.shape, big_tensor.shape)

    # Check the value of the tensor
    torch.testing.assert_close(local_tensor, big_tensor, check_device=False)

  def test_to_local_requires_grad(self):
    """Test that gradients flow correctly through to_local()."""
    # Create a tensor with requires_grad=True
    world_size = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", list(range(world_size)))

    tensor = torch.randn(100_000, 88, requires_grad=True)

    # Create XLAShardedTensor
    sharded_tensor = XLAShardedTensor(tensor, mesh, [Shard(0)])

    # Verify requires_grad is set
    self.assertTrue(sharded_tensor.requires_grad)

    res = sharded_tensor.sum()
    res.backward()

    # Verify grad are calculated
    self.assertTrue(sharded_tensor.grad is not None)

    # Call to local function
    local_tensor = sharded_tensor.to_local()

    # Verify requires_grad is preserved
    self.assertTrue(local_tensor.requires_grad)

    # All gradients should be 1.0 since we did a sum()
    self.assertTrue(torch.allclose(local_tensor.grad, torch.ones_like(tensor)))

    print("Gradient flow test successful")


if __name__ == "__main__":
  result = unittest.main(exit=False)
  sys.exit(0 if result.result.wasSuccessful() else 1)
