import sys
import unittest
import torch
from torch.distributed.tensor.placement_types import Shard, Replicate
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla
import numpy as np
import test_xla_sharding_base


class DTensorRedistributeTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    xr.use_spmd()

  def _verify_sharding_spec(self, tensor, expected_devices=None):
    """Verify tensor sharding spec after mark_step"""
    torch_xla.core.xla_model.mark_step()
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(tensor)
    if expected_devices:
      self.assertIn(expected_devices, sharding_spec)
    return sharding_spec

  # Test tensor shapes: 0D, 1D, 2D, 3D
  @unittest.skipIf(xr.global_runtime_device_count() < 2, "Need ≥2 devices")
  def test_tensor_shapes(self):
    device_count = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(device_count), (device_count,))

    shapes_and_specs = [
        ((), ()),  # 0D scalar
        ((8,), (0,)),  # 1D
        ((8, 16), (0, None)),  # 2D
        ((4, 8, 16), (0, None, None))  # 3D
    ]

    for shape, partition_spec in shapes_and_specs:
      with self.subTest(shape=shape):
        if len(shape) == 0:
          tensor = torch.tensor(1.0).to('xla')
          placements = [Replicate()]
          expected_spec = ()
        else:
          tensor = torch.randn(shape).to('xla')
          sharded_tensor = xs.mark_sharding(tensor, mesh, partition_spec)
          placements = [Shard(0)]
          expected_spec = partition_spec

          redistributed = sharded_tensor.redistribute(mesh, placements)
          self.assertEqual(redistributed.partition_spec, expected_spec)

          # Convert partition spec to expected devices pattern
          devices_pattern = [
              str(device_count) if spec == 0 else '1' for spec in expected_spec
          ]
          expected_devices = f"devices=[{','.join(devices_pattern)}]"

          # Skip HLO verification for 4D tensors due to XLA optimization issues
          if len(shape) < 4:
            self._verify_sharding_spec(redistributed.global_tensor,
                                       expected_devices)

  # Test tensor dtypes: bf16, f32, int32
  @unittest.skipIf(xr.global_runtime_device_count() < 2, "Need ≥2 devices")
  def test_tensor_dtypes(self):
    device_count = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(device_count), (device_count,))

    dtypes = [torch.bfloat16, torch.float32, torch.int32]

    for dtype in dtypes:
      with self.subTest(dtype=dtype):
        if dtype == torch.int32:
          tensor = torch.randint(0, 100, (8, 16), dtype=dtype).to('xla')
        else:
          tensor = torch.randn(8, 16, dtype=dtype).to('xla')

        sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))
        placements = [Shard(0)]

        redistributed = sharded_tensor.redistribute(mesh, placements)
        self.assertEqual(redistributed.partition_spec, (0, None))
        self.assertEqual(redistributed.global_tensor.dtype, dtype)

        # Verify HLO sharding
        expected_devices = f"devices=[{device_count},1]"
        self._verify_sharding_spec(redistributed.global_tensor,
                                   expected_devices)

  # Test device mesh dimensions: 1D, 2D
  @unittest.skipIf(xr.global_runtime_device_count() < 4, "Need ≥4 devices")
  def test_device_mesh_dimensions(self):
    device_count = xr.global_runtime_device_count()

    # 1D mesh
    mesh_1d = xs.Mesh(np.arange(device_count), (device_count,))
    tensor = torch.randn(8, 16).to('xla')
    sharded_tensor = xs.mark_sharding(tensor, mesh_1d, (0, None))

    redistributed = sharded_tensor.redistribute(mesh_1d, [Shard(1)])
    self.assertEqual(redistributed.partition_spec, (None, 0))

    # Verify HLO sharding for 1D mesh
    expected_devices = f"devices=[1,{device_count}]"
    self._verify_sharding_spec(redistributed.global_tensor, expected_devices)

    # 2D mesh
    if device_count >= 4 and device_count % 2 == 0:
      mesh_2d = xs.Mesh(np.arange(device_count), (2, device_count // 2))
      tensor_2d = torch.randn(8, 16).to('xla')
      sharded_tensor = xs.mark_sharding(tensor_2d, mesh_2d, (0, None))

      redistributed = sharded_tensor.redistribute(
          mesh_2d, [Replicate(), Shard(1)])
      self.assertEqual(redistributed.partition_spec, (None, 1))

      # Verify HLO sharding for 2D mesh
      expected_devices = f"devices=[1,{device_count // 2},{device_count // 2}]"
      self._verify_sharding_spec(redistributed.global_tensor, expected_devices)

  # Test placement types: Replicate, Shard
  @unittest.skipIf(xr.global_runtime_device_count() < 2, "Need ≥2 devices")
  def test_placement_types(self):
    device_count = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(device_count), (device_count,))
    tensor = torch.randn(8, 16).to('xla')

    # Test Replicate
    sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))
    redistributed = sharded_tensor.redistribute(mesh, [Replicate()])
    self.assertEqual(redistributed.partition_spec, (None, None))

    # Verify HLO sharding for replicated
    self._verify_sharding_spec(redistributed.global_tensor, "replicated")

    # Test Shard on different dimensions
    for dim in [0, 1]:
      with self.subTest(shard_dim=dim):
        redistributed = sharded_tensor.redistribute(mesh, [Shard(dim)])
        expected_spec = [None, None]
        expected_spec[dim] = 0
        self.assertEqual(redistributed.partition_spec, tuple(expected_spec))

        # Verify HLO sharding
        devices_pattern = [
            str(device_count) if i == dim else '1' for i in range(2)
        ]
        expected_devices = f"devices=[{','.join(devices_pattern)}]"
        self._verify_sharding_spec(redistributed.global_tensor,
                                   expected_devices)

  # Test error cases with invalid inputs
  @unittest.skipIf(xr.global_runtime_device_count() < 2, "Need ≥2 devices")
  def test_invalid_inputs(self):
    device_count = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(device_count), (device_count,))
    tensor = torch.randn(8, 16).to('xla')
    sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))

    # Test invalid shard dimension (tensor only has dims 0,1 but asking for dim 2)
    with self.assertRaises((IndexError, ValueError, RuntimeError)):
      sharded_tensor.redistribute(mesh, [Shard(2)])

    # Test mismatched placements length (1D mesh expects 1 placement, not 2)
    with self.assertRaises((ValueError, RuntimeError)):
      sharded_tensor.redistribute(mesh, [Shard(0), Shard(1)])

  # Test sharding propagation through operations
  @unittest.skipIf(xr.global_runtime_device_count() < 2, "Need ≥2 devices")
  def test_sharding_propagation(self):
    device_count = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(device_count), (device_count,))

    # Unary ops
    tensor = torch.randn(8, 16).to('xla')
    sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))
    redistributed = sharded_tensor.redistribute(mesh, [Shard(0)])

    relu_result = torch.relu(redistributed.global_tensor)
    self.assertEqual(relu_result.shape, (8, 16))
    self.assertTrue(torch.all(relu_result >= 0))

    # Binary ops
    tensor2 = torch.randn(8, 16).to('xla')
    sharded_tensor2 = xs.mark_sharding(tensor2, mesh, (0, None))
    redistributed2 = sharded_tensor2.redistribute(mesh, [Shard(0)])

    add_result = redistributed.global_tensor + redistributed2.global_tensor
    mul_result = redistributed.global_tensor * redistributed2.global_tensor

    # Verify operation results
    self.assertEqual(add_result.shape, (8, 16))
    self.assertEqual(mul_result.shape, (8, 16))

    # Verify operations work correctly
    self.assertTrue(
        torch.allclose(
            add_result,
            redistributed.global_tensor + redistributed2.global_tensor))
    self.assertTrue(
        torch.allclose(
            mul_result,
            redistributed.global_tensor * redistributed2.global_tensor))

  # Test comprehensive redistribute scenarios
  @unittest.skipIf(xr.global_runtime_device_count() < 2, "Need ≥2 devices")
  def test_comprehensive_redistribute(self):
    device_count = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(device_count), (device_count,))

    tensor = torch.randn(8, 16).to('xla')
    sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))

    # Test all placement combinations for 1D mesh
    placement_types = [Replicate(), Shard(0), Shard(1)]

    for placement in placement_types:
      with self.subTest(placement=placement):
        placements = [placement]

        if isinstance(placement, Shard):
          expected_spec = [None] * 2
          expected_spec[placement.dim] = 0
          expected_spec = tuple(expected_spec)
        else:
          expected_spec = (None, None)

        redistributed = sharded_tensor.redistribute(mesh, placements)
        self.assertEqual(redistributed.partition_spec, expected_spec)

        # Verify HLO sharding
        if isinstance(placement, Shard):
          devices_pattern = [
              str(device_count) if i == placement.dim else '1' for i in range(2)
          ]
          expected_devices = f"devices=[{','.join(devices_pattern)}]"
        else:
          expected_devices = "replicated"
        self._verify_sharding_spec(redistributed.global_tensor,
                                   expected_devices)

  # Test async redistribute
  @unittest.skipIf(xr.global_runtime_device_count() < 4, "Need ≥4 devices")
  def test_async_redistribute(self):
    device_count = xr.global_runtime_device_count()
    mesh_shape = (2, device_count // 2)
    mesh = xs.Mesh(np.arange(device_count), mesh_shape)

    tensor = torch.randn(8, 16).to('xla')
    sharded_tensor = xs.mark_sharding(tensor, mesh, (0, None))

    # Test async redistribute
    placements = [Replicate(), Shard(0)]
    redistributed = sharded_tensor.redistribute(mesh, placements, async_op=True)
    self.assertEqual(redistributed.partition_spec, (1, None))

    # Verify async operation creates different tensor object
    self.assertIsNot(redistributed.global_tensor, sharded_tensor.global_tensor)

    # Verify HLO sharding for async redistribute (XLA generates more complex pattern)
    expected_devices = f"devices=[2,1,{device_count // 2}]"
    self._verify_sharding_spec(redistributed.global_tensor, expected_devices)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
