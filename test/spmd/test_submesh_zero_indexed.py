import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import torch_xla.core.xla_env_vars as xenv
import torch_xla.utils.utils as xu
import sys

import test_xla_sharding_base


class SubmeshZeroIndexedTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def _create_zero_indexed_submesh_2dev(self):
    """Create 2-device submesh starting from device 0: [0,1]"""
    device_ids = [0, 1]
    mesh_shape = (1, 2)
    axis_names = ('x', 'y')
    return Mesh(device_ids, mesh_shape, axis_names)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern1_2dev(self):
    """shard both tensors -> compute -> cpu() -> sync()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    t1 = torch.randn(4, 4, device='cpu')
    t2 = torch.randn(4, 4, device='cpu')
    expected = torch.matmul(t1, t2)

    xt1 = t1.to(torch_xla.device())
    xt2 = t2.to(torch_xla.device())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    result_cpu = result.cpu()
    torch_xla.sync()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern1_2dev_direct_device(self):
    """direct device creation: shard both tensors -> compute -> cpu() -> sync()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu(), xt2.cpu())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    result_cpu = result.cpu()
    torch_xla.sync()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern2_2dev(self):
    """shard both tensors -> compute -> sync() -> cpu()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    t1 = torch.randn(4, 4, device='cpu')
    t2 = torch.randn(4, 4, device='cpu')
    expected = torch.matmul(t1, t2)

    xt1 = t1.to(torch_xla.device())
    xt2 = t2.to(torch_xla.device())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern2_2dev_direct_device(self):
    """direct device creation: shard both tensors -> compute -> sync() -> cpu()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu(), xt2.cpu())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern3_2dev(self):
    """shard one tensor -> compute -> cpu() -> sync()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    t1 = torch.randn(4, 4, device='cpu')
    t2 = torch.randn(4, 4, device='cpu')
    expected = torch.matmul(t1, t2)

    xt1 = t1.to(torch_xla.device())
    xt2 = t2.to(torch_xla.device())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    result_cpu = result.cpu()
    torch_xla.sync()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern3_2dev_direct_device(self):
    """direct device creation: shard one tensor -> compute -> cpu() -> sync()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu(), xt2.cpu())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    result_cpu = result.cpu()
    torch_xla.sync()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern4_2dev_direct_device(self):
    """direct device creation: shard one tensor -> compute -> sync() -> cpu()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu(), xt2.cpu())

    xs.mark_sharding(xt1, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern5_2dev_direct_device(self):
    """direct device creation: modify tensor -> shard one tensor -> compute -> sync() -> cpu()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu() + 2, xt2.cpu())

    xt1 += 2
    xs.mark_sharding(xt1, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern6_2dev_direct_device(self):
    """direct device creation: modify tensor -> shard both tensors -> compute -> sync() -> cpu()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu() + 2, xt2.cpu())

    xt1 += 2
    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_pattern7_2dev_direct_device(self):
    """direct device creation: modify tensor -> shard one tensor -> compute -> cpu() -> sync()"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    expected = torch.matmul(xt1.cpu() + 2, xt2.cpu())

    xt1 += 2
    xs.mark_sharding(xt1, mesh, ('x', 'y'))

    result = torch.matmul(xt1, xt2)
    result_cpu = result.cpu()
    torch_xla.sync()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_single_tensor_addition_2dev(self):
    """Single tensor addition test with zero-indexed 2-device submesh"""
    mesh = self._create_zero_indexed_submesh_2dev()

    t = torch.randn(4, 4, device='cpu')
    expected = t + 4.2

    xt = t.to(torch_xla.device())
    xs.mark_sharding(xt, mesh, ('x', None))

    result = xt + 4.2
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for 2-device submesh tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_single_tensor_addition_2dev_direct_device(self):
    """Single tensor addition test with direct device creation and zero-indexed 2-device submesh"""
    mesh = self._create_zero_indexed_submesh_2dev()

    xt = torch.randn(4, 4, device=torch_xla.device())
    expected = xt.cpu() + 4.2

    xs.mark_sharding(xt, mesh, ('x', None))

    result = xt + 4.2
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(
      xr.global_runtime_device_count() >= 4,
      "Need at least 4 devices for advanced direct device tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_complex_operations_direct_device(self):
    """Test complex tensor operations with direct device creation"""
    mesh = self._create_zero_indexed_submesh_2dev()

    # Create tensors directly on device
    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())
    xt3 = torch.randn(4, 4, device=torch_xla.device())

    # Create expected result on CPU for comparison
    t1_cpu = xt1.cpu()
    t2_cpu = xt2.cpu()
    t3_cpu = xt3.cpu()
    expected = torch.matmul(t1_cpu + t2_cpu, t3_cpu)

    # Apply sharding
    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))
    xs.mark_sharding(xt3, mesh, ('x', 'y'))

    # Perform complex operations
    result = torch.matmul(xt1 + xt2, xt3)
    torch_xla.sync()
    result_cpu = result.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  @unittest.skipUnless(
      xr.global_runtime_device_count() >= 4,
      "Need at least 4 devices for advanced direct device tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU", "Submesh tests require CPU")
  def test_inplace_operations_direct_device(self):
    """Test in-place operations with direct device creation"""
    mesh = self._create_zero_indexed_submesh_2dev()

    # Create tensors directly on device
    xt1 = torch.randn(4, 4, device=torch_xla.device())
    xt2 = torch.randn(4, 4, device=torch_xla.device())

    # Store original values for expected calculation
    t1_orig = xt1.cpu().clone()
    t2_orig = xt2.cpu().clone()

    # Apply sharding
    xs.mark_sharding(xt1, mesh, ('x', 'y'))
    xs.mark_sharding(xt2, mesh, ('x', 'y'))

    # Perform in-place operations
    xt1 *= 2.0
    xt1 += xt2

    # Calculate expected result
    expected = t1_orig * 2.0 + t2_orig

    torch_xla.sync()
    result_cpu = xt1.cpu()

    self.assertTrue(torch.allclose(expected, result_cpu, atol=1e-5))

  # Error validation test cases for xs.Mesh constructor
  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for error validation tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU",
      "Error validation tests require CPU")
  def test_mesh_axis_names_length_mismatch(self):
    """Test error when axis names length doesn't match mesh dimensions"""
    device_ids = [0, 1]
    mesh_shape = (1, 2)  # 2 dimensions
    axis_names = ('data', 'model', 'extra')  # 3 names - mismatch!

    with self.assertRaisesRegex(
        AssertionError, "Number of axis names .* must match mesh dimensions"):
      Mesh(device_ids, mesh_shape, axis_names)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for error validation tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU",
      "Error validation tests require CPU")
  def test_mesh_duplicate_axis_names(self):
    """Test error when axis names are not unique"""
    device_ids = [0, 1]
    mesh_shape = (1, 2)
    axis_names = ('data', 'data')  # Duplicate names!

    with self.assertRaisesRegex(AssertionError, "Axis names must be unique"):
      Mesh(device_ids, mesh_shape, axis_names)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for error validation tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU",
      "Error validation tests require CPU")
  def test_mesh_device_count_mismatch(self):
    """Test error when device IDs count doesn't match mesh size"""
    device_ids = [0, 1]  # 2 devices
    mesh_shape = (2, 2)  # mesh size = 4, but only 2 device IDs!

    with self.assertRaisesRegex(AssertionError,
                                "Number of device IDs .* must match mesh size"):
      Mesh(device_ids, mesh_shape)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for error validation tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU",
      "Error validation tests require CPU")
  def test_mesh_duplicate_device_ids(self):
    """Test error when device IDs are not unique"""
    device_ids = [0, 0]  # Duplicate device IDs!
    mesh_shape = (1, 2)

    with self.assertRaisesRegex(AssertionError, "Device IDs must be unique"):
      Mesh(device_ids, mesh_shape)

  @unittest.skipUnless(xr.global_runtime_device_count() >= 4,
                       "Need at least 4 devices for error validation tests")
  @unittest.skipUnless(
      xu.getenv_as(xenv.PJRT_DEVICE, str) == "CPU",
      "Error validation tests require CPU")
  def test_mesh_device_ids_out_of_bounds(self):
    """Test error when device IDs are outside addressable device range"""
    # Assuming we have 4 devices, use invalid IDs like 10,11
    device_ids = [10, 11]  # Out of bounds device IDs!
    mesh_shape = (1, 2)

    with self.assertRaisesRegex(
        AssertionError,
        "Device IDs has to be subset of addressable_devices; got:*."):
      Mesh(device_ids, mesh_shape)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
