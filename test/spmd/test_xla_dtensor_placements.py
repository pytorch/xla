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

  def test_placements_basic(self):
    """Test that placements property works when XLAShardedTensor is properly initialized."""

    world_size = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", torch.arange(world_size))
    big_tensor = torch.randn(100_000, 88)

    # Create a sharded tensor with placements
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])

    # Test that placements property works on XLAShardedTensor
    assert hasattr(
        my_dtensor,
        'placements'), "XLAShardedTensor should have placements property"
    assert my_dtensor.placements == (
        Shard(0),), f"Expected (Shard(0),), got {my_dtensor.placements}"

  def test_placements_failure(self):
    """Test that placements property provides a helpful error message when sharding info is missing."""
    big_tensor = torch.randn(100_000, 88)

    # Create XLAShardedTensor without sharding information
    xla_tensor = XLAShardedTensor(big_tensor)

    # Test that accessing placements raises the expected error
    with self.assertRaises(ValueError) as context:
      _ = xla_tensor.placements

    expected_message = (
        "Placements not available: XLAShardedTensor requires mesh_shape and "
        "partition_spec to be set. Use mark_sharding() to properly initialize sharding information."
    )
    self.assertEqual(
        str(context.exception), expected_message,
        "Error message should match exactly for user clarity")

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

    # Invalidate cache and verify third access
    def invalidate_spec_cach(tensor):
      tensor._cached_spec = None

    invalidate_spec_cach(my_dtensor)
    self.assertIsNone(my_dtensor._cached_spec,
                      "Cache should be None after invalidation")

    # Third access
    placements3 = my_dtensor.placements
    self.assertIsNotNone(my_dtensor._cached_spec,
                         "Cache should be recreated after invalidation")
    self.assertEqual(placements3, (Replicate(),),
                     "New cache should have correct placements")


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
