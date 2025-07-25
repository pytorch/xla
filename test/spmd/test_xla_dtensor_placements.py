import os
import sys

import torch
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate

import torch_xla
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import XLAShardedTensor
from torch_xla.distributed.spmd.xla_sharding import wrap_as_sharded_tensor

import unittest
import test_xla_sharding_base


class XLADTensorSpecConversionTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_placements_property_success(self):
    """Test that placements property works when XLAShardedTensor is properly initialized."""

    world_size = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", torch.arange(world_size))
    big_tensor = torch.randn(100_000, 88)
    
    # Create properly sharded tensor using distribute_tensor
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])
    
    # Test that placements property works on XLAShardedTensor
    assert hasattr(
        my_dtensor,
        'placements'), "XLAShardedTensor should have placements property"
    assert my_dtensor.placements == (
        Shard(0),), f"Expected (Shard(0),), got {my_dtensor.placements}"
    
    print(
      "Placements property works correctly for properly initialized XLAShardedTensor"
    )

  def test_placements_property_failure(self):
    """Test that placements property fails with proper error when XLAShardedTensor lacks sharding info."""
    big_tensor = torch.randn(100_000, 88)
    
    # Create XLAShardedTensor without proper sharding information
    # This creates a tensor without mesh_shape or partition_spec
    xla_tensor = XLAShardedTensor(big_tensor)
    
    # Test that accessing placements raises the expected error
    with self.assertRaises(ValueError) as context:
      _ = xla_tensor.placements
    
    # Verify the error message is appropriate
    error_message = str(context.exception)
    self.assertIn(
      "placements", error_message.lower(), 
      f"Error message should mention 'placements': {error_message}")
    
    # Check for specific error indicators
    expected_keywords = ["mesh_shape", "partition_spec", "sharding"]
    has_expected_keyword = any(
        keyword in error_message.lower() for keyword in expected_keywords)
    self.assertTrue(
        has_expected_keyword, 
        f"Error message should mention sharding concepts: {error_message}")
    
    print(f"Placements property correctly fails with error: {error_message}")

  def test_placements_caching_behavior(self):
    """Test that placements property uses caching correctly."""
    world_size = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", torch.arange(world_size))
    big_tensor = torch.randn(100_000, 88)
    
    # Create properly sharded tensor
    my_dtensor = distribute_tensor(big_tensor, mesh, [Replicate()])
    
    # First access should create the cache
    placements1 = my_dtensor.placements
    self.assertIsNotNone(my_dtensor._cached_spec, 
                         "Cache should be created after first access")
    
    # Second access should use cache
    placements2 = my_dtensor.placements
    self.assertEqual(placements1, placements2, 
                     "Cached placements should be identical")
    self.assertEqual(placements1, (Replicate(),), 
                     f"Expected (Replicate(),), got {placements1}")
    
    print("Placements property caching works correctly")


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)


