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

  def test_sample_test_case(self):
    world_size = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", torch.arange(world_size))
    big_tensor = torch.randn(100000, 88)
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])

    assert my_dtensor._spec.mesh.device_type == mesh.device_type
    assert my_dtensor._spec.placements == (Shard(0),)

  def test_xla_to_dtensor_spec_conversion(self):
    device_count = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", list(range(device_count)))

    # Test different sharding patterns
    test_cases = [
        (torch.randn(100, 50), [Shard(0)]),
        (torch.randn(100, 50), [Shard(1)]),
        (torch.randn(100, 50, 25), [Shard(0)]),
        (torch.randn(100, 50), [Replicate()]),
    ]

    for tensor, placements in test_cases:
      xla_tensor = distribute_tensor(tensor, mesh, placements)
      spec = xla_tensor._spec

      assert spec is not None
      assert spec.mesh.device_type == "xla"
      assert spec.tensor_meta.shape == tensor.shape
      assert spec.tensor_meta.dtype == tensor.dtype
      assert len(spec.placements) >= 1
      assert spec.placements == tuple(placements)

  def test_mesh_conversion(self):
    device_count = xr.global_runtime_device_count()
    original_mesh = DeviceMesh("xla", list(range(device_count)))
    tensor = torch.randn(50, 50)
    xla_tensor = distribute_tensor(tensor, original_mesh, [Shard(0)])

    converted_spec = xla_tensor._spec

    assert converted_spec.mesh.device_type == "xla"
    assert converted_spec.mesh.size() == device_count
    # assert on mesh dimensions
    assert converted_spec.mesh.shape == original_mesh.shape

  def test_spec_caching(self):
    """Test that _spec property caches results
    """
    device_count = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", list(range(device_count)))
    tensor = torch.randn(100, 100)
    xla_tensor = distribute_tensor(tensor, mesh, [Shard(0)])

    spec1 = xla_tensor._spec

    assert xla_tensor._cached_spec is not None
    assert xla_tensor._cached_spec is spec1

    spec2 = xla_tensor._spec
    assert spec1 is spec2

  def _create_test_tensor_and_mesh(self, tensor_shape, mesh_shape, placements):
    """Helper to create tensor and mesh for testing"""
    device_count = xr.global_runtime_device_count()
    if device_count < max(mesh_shape):
      self.skipTest(
          f"Need at least {max(mesh_shape)} devices, got {device_count}")

    mesh = DeviceMesh("xla", torch.arange(device_count).reshape(mesh_shape))
    tensor = torch.randn(*tensor_shape)
    return distribute_tensor(tensor, mesh, placements), mesh

  def test_multi_dim_sharding_spec(self):
    """Test _spec for multi-dimensional sharding"""
    device_count = xr.global_runtime_device_count()
    if device_count < 4:
      self.skipTest("Need at least 4 devices for 2D mesh")

    mesh_shape = (2, device_count // 2)
    xla_tensor, mesh = self._create_test_tensor_and_mesh(
        (100, 50), mesh_shape, [Shard(0), Shard(1)])
    spec = xla_tensor._spec

    assert len(spec.placements) == 2
    assert spec.mesh.ndim == 2

  def test_mixed_placement_spec(self):
    """Test _spec for tensors with mixed shard/replicate placements"""
    device_count = xr.global_runtime_device_count()
    if device_count < 4:
      self.skipTest("Need at least 4 devices for 2D mesh")

    mesh_shape = (2, device_count // 2)
    xla_tensor, mesh = self._create_test_tensor_and_mesh(
        (100, 50), mesh_shape, [Shard(0), Replicate()])
    spec = xla_tensor._spec

    assert len(spec.placements) == 2
    assert isinstance(spec.placements[0], Shard)
    assert isinstance(spec.placements[1], Replicate)

  def test_sharding_info_acquisition(self):
    """Test that non-XLAShardedTensor can acquire sharding information

    Tests case of 'elem is not an XLAShardedTensor but there exists 
    sharding information we want to acquire'
    """

    device_count = xr.global_runtime_device_count()
    mesh_shape = (device_count,)
    partition_spec = (0, None)

    regular_tensor = torch.randn(100, 50).to('xla')

    sharded_tensor = wrap_as_sharded_tensor(
        regular_tensor, mesh_shape=mesh_shape, partition_spec=partition_spec)

    # Verify the tensor acquired the sharding information
    assert isinstance(sharded_tensor, XLAShardedTensor)
    assert sharded_tensor.mesh_shape == mesh_shape
    assert sharded_tensor.partition_spec == partition_spec

  def test_resharding_logic(self):
    """
    Tests wrap_as_sharded_tensor resharding before returning XLAShardedTensor t.
    """

    device_count = xr.global_runtime_device_count()
    if device_count < 4:
      self.skipTest("Need at least 4 devices for resharding test")

    # Initial sharding
    initial_mesh_shape = (device_count,)
    initial_partition_spec = (0, None)
    new_mesh_shape = (2, device_count // 2)
    new_partition_spec = (0, 1)

    # Create tensor and verify resharding
    tensor = torch.randn(100, 50).to('xla')
    sharded_tensor = wrap_as_sharded_tensor(
        tensor,
        mesh_shape=initial_mesh_shape,
        partition_spec=initial_partition_spec)
    initial_spec = sharded_tensor._spec

    resharded_tensor = wrap_as_sharded_tensor(
        sharded_tensor,
        mesh_shape=new_mesh_shape,
        partition_spec=new_partition_spec)

    # Verify resharding worked and cache was invalidated
    assert resharded_tensor.mesh_shape == new_mesh_shape
    assert resharded_tensor.partition_spec == new_partition_spec
    assert resharded_tensor._spec is not initial_spec

  def test_spec_invalidation_on_resharding(self):
    """Tests cases where the cached spec may become outdated.
    """

    device_count = xr.global_runtime_device_count()
    if device_count < 4:
      self.skipTest("Need at least 4 devices for resharding test")

    tensor = torch.randn(100, 50).to('xla')
    initial_mesh_shape = (device_count,)
    initial_partition_spec = (0, None)
    new_mesh_shape = (2, device_count // 2)
    new_partition_spec = (0, 1)

    sharded_tensor = wrap_as_sharded_tensor(
        tensor,
        mesh_shape=initial_mesh_shape,
        partition_spec=initial_partition_spec)
    initial_spec = sharded_tensor._spec
    assert sharded_tensor._cached_spec is not None

    # Changing mesh_shape / partition_spec through wrap_as_sharded_tensor invalidates cache
    resharded_tensor = wrap_as_sharded_tensor(
        sharded_tensor,
        mesh_shape=new_mesh_shape,
        partition_spec=initial_partition_spec)
    assert resharded_tensor._spec is not initial_spec
    assert resharded_tensor._spec.mesh.shape == new_mesh_shape

    initial_spec = resharded_tensor._spec
    resharded_tensor = wrap_as_sharded_tensor(
        resharded_tensor,
        mesh_shape=new_mesh_shape,
        partition_spec=new_partition_spec)
    assert resharded_tensor._spec is not initial_spec
    assert resharded_tensor._spec.placements[1].dim == 1

  def test_auto_wrapped_tensor_spec_failure(self):
    """Test that auto-wrapped tensors fail when accessing _spec property.
    
    Auto-wrapped tensors are created through operations that trigger __torch_dispatch__
    but don't yet have access to the sharding propagation done through open xla,
    causing ._spec to fail. 
    """
    device_count = xr.global_runtime_device_count()
    mesh = DeviceMesh("xla", torch.arange(device_count))
    tensor = torch.randn(4, 4)
    sharded_tensor = distribute_tensor(tensor, mesh, [Shard(0)])

    auto_wrapped = sharded_tensor + sharded_tensor

    with self.assertRaises(ValueError):
      _ = auto_wrapped._spec


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
