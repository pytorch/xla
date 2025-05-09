import os
import sys
import unittest

import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr
from torch_xla.experimental.assume_pure import assume_pure
from torch_xla.distributed.spmd import mark_sharding, mark_sharding_with_gradients, set_global_mesh, get_1d_mesh, Mesh


class AssumePureSpmdTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Activate SPMD
    xr.use_spmd()

  def setUp(self):
    # Set up a simple SPMD mesh for these tests.
    self.spmd_mesh = get_1d_mesh(axis_name="model")
    set_global_mesh(self.spmd_mesh)

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required")
  @unittest.skipIf(
      torch.cuda.is_available() or os.environ.get('PJRT_DEVICE') == 'CUDA',
      "TODO(https://github.com/pytorch/xla/issues/9017): Get these tests working on GPU"
  )
  def test_assume_pure_works_with_mark_sharding(self):
    x = torch.randn((8, 4, 5, 128), device='xla')
    result = assume_pure(mark_sharding)(x, self.spmd_mesh,
                                        ("model", None, None, None))
    torch_xla.sync(wait=True)
    N = xr.global_runtime_device_count()
    self.assertIn(f'devices=[{N}',
                  torch_xla._XLAC._get_xla_sharding_spec(result))

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required")
  @unittest.skipIf(
      torch.cuda.is_available() or os.environ.get('PJRT_DEVICE') == 'CUDA',
      "TODO(https://github.com/pytorch/xla/issues/9017): Get these tests working on GPU"
  )
  def test_assume_pure_works_with_mark_sharding_with_gradients(self):
    x = torch.randn((8, 4, 5, 128)).to('xla').requires_grad_(True)
    result = assume_pure(mark_sharding_with_gradients)(
        x, self.spmd_mesh, ("model", None, None, None))
    result.sum().backward()
    torch_xla.sync(wait=True)
    N = xr.global_runtime_device_count()
    self.assertIn(f'devices=[{N}',
                  torch_xla._XLAC._get_xla_sharding_spec(result))
    assert x.grad is not None
    self.assertIn(f'devices=[{N}',
                  torch_xla._XLAC._get_xla_sharding_spec(x.grad))

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required")
  @unittest.skipIf(
      torch.cuda.is_available() or os.environ.get('PJRT_DEVICE') == 'CUDA',
      "TODO(https://github.com/pytorch/xla/issues/9017): Get these tests working on GPU"
  )
  def test_convert_to_jax_mesh(self):
    jax_mesh = self.spmd_mesh.get_jax_mesh()
    self.assertEqual(jax_mesh.devices.shape, self.spmd_mesh.mesh_shape)
    np.testing.assert_equal(
        np.array([dev.id for dev in jax_mesh.devices.flatten()]),
        self.spmd_mesh.device_ids)

  @unittest.skipUnless(xr.global_runtime_device_count() > 1,
                       "Multiple devices required")
  @unittest.skipUnless(os.environ.get('PJRT_DEVICE') == 'TPU', "TPU only test")
  def test_convert_to_jax_mesh_shuffled(self):
    """Test get_jax_mesh when the PyTorch/XLA mesh has a custom order."""

    # Arrange
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    device_ids = np.random.permutation(device_ids)
    self.spmd_mesh = Mesh(
        device_ids, mesh_shape=(num_devices,), axis_names=('model',))

    # Act
    jax_mesh = self.spmd_mesh.get_jax_mesh()

    # Assert
    torch_xla_devices = np.array(
        [xr.global_runtime_device_attributes()[i] for i in device_ids])
    self.assertEqual(jax_mesh.devices.shape, self.spmd_mesh.mesh_shape)
    np.testing.assert_equal(
        np.array([dev.coords for dev in jax_mesh.devices.flatten()]),
        np.array([dev['coords'] for dev in torch_xla_devices.flatten()]),
    )


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
