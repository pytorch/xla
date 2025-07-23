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

    def test_basic_conversion(self):
        """Test basic conversion of regular tensor to XLAShardedTensor."""
        world_size = xr.global_runtime_device_count()
        
        # Create a regular tensor (not on XLA device)
        tensor = torch.randn(100_000, 88)
        tensor_cpu = tensor.cpu()  # Keep a CPU copy for comparison
        
        # Create a DeviceMesh
        device_mesh = DeviceMesh("xla", list(range(world_size)))
        
        # Use DTensor.from_local with the regular tensor
        dt = DTensor.from_local(tensor, device_mesh=device_mesh)
        
        # Verify the tensor was converted correctly
        self.assertEqual(dt.shape, tensor.shape)
        
        # Check the value of the tensor
        torch.testing.assert_close(dt.global_tensor, tensor_cpu, check_device=False)
        
        # Verify operations work
        result = dt + 1.0
        self.assertEqual(result.shape, tensor.shape)
        
        print("Basic conversion successful")


    def test_conversion_with_placements(self):
        """Test conversion with explicit placements."""
        world_size = xr.global_runtime_device_count()
        
        # Create a regular tensor (not on XLA device)
        tensor = torch.randn(100_000, 88)
        tensor_cpu = tensor.cpu()  # Keep a CPU copy for comparison
        
        # Create a DeviceMesh
        device_mesh = DeviceMesh("xla", list(range(world_size)))
        
        # Use DTensor.from_local with explicit placements
        dt = DTensor.from_local(
            tensor,
            device_mesh=device_mesh,
            placements=[Replicate()]
        )
        
        # Verify the tensor was converted correctly
        self.assertEqual(dt.shape, tensor.shape)

        # Check the value of the tensor
        torch.testing.assert_close(dt.global_tensor, tensor_cpu, check_device=False)
        
        # Verify operations work
        result = dt + 1.0
        self.assertEqual(result.shape, tensor.shape)
        
        print("Conversion with placements successful")

    def test_conversion_with_sharding(self):
        """Test conversion with sharding placement."""
        world_size = xr.global_runtime_device_count()
        if world_size < 2:
            self.skipTest("Need at least 2 devices for sharding test")
        
        # Create a tensor divisible by world_size
        tensor = torch.randn(100_000, 88)
        tensor_cpu = tensor.cpu()  # Keep a CPU copy for comparison
        
        # Create a DeviceMesh
        device_mesh = DeviceMesh("xla", list(range(world_size)))
        
        # Use DTensor.from_local with sharding placement
        dt = DTensor.from_local(
            tensor,
            device_mesh=device_mesh,
            placements=[Shard(0)]
        )
        
        # Verify the tensor was converted correctly
        self.assertEqual(dt.shape, tensor.shape)

        # Check the value of the tensor
        torch.testing.assert_close(dt.global_tensor, tensor_cpu, check_device=False)
        
        # Verify operations work
        result = dt + 1.0
        self.assertEqual(result.shape, tensor.shape)
        
        print("Conversion with sharding successful")

    def test_conversion_with_different_dtypes(self):
        """Test conversion with different dtypes."""
        world_size = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(world_size)))
        
        # Test with different dtypes
        for dtype in [torch.float16, torch.float32, torch.int32, torch.int64]:
            # Create a tensor with specific dtype
            tensor = torch.ones(100_000, 88, dtype=dtype)
            tensor_cpu = tensor.cpu()  # Keep a CPU copy for comparison
            
            # Use DTensor.from_local with the tensor
            dt = DTensor.from_local(tensor, device_mesh=device_mesh)
            
            # Verify dtype is preserved
            self.assertEqual(dt.dtype, dtype)
            
            # Check the value of the tensor
            torch.testing.assert_close(dt.global_tensor, tensor_cpu, check_device=False)
            
            # Verify operations work
            if dtype.is_floating_point:
                result = dt + 1.0
            else:
                result = dt + 1
                
            self.assertEqual(result.shape, tensor.shape)
            self.assertEqual(result.dtype, dtype)
            
            print(f"Conversion with {dtype} successful")


if __name__ == "__main__":
    result = unittest.main(exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)