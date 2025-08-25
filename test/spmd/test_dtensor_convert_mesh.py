import sys
import unittest
import torch
from torch.distributed.tensor import DeviceMesh, init_device_mesh
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from torch_xla.distributed.spmd.api import convert_to_xla_mesh
import test_xla_sharding_base


class ConvertToXlaMeshIntegrationTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed for 1D mesh test")
  def test_convert_1d_device_mesh(self):
    device_count = xr.global_runtime_device_count()
    dt_mesh = init_device_mesh("xla", mesh_shape=(device_count,))

    xla_mesh = convert_to_xla_mesh(dt_mesh)

    self.assertIsInstance(xla_mesh, Mesh)
    self.assertEqual(len(xla_mesh.device_ids), device_count)
    self.assertEqual(xla_mesh.mesh_shape, (device_count,))
    self.assertEqual(xla_mesh.axis_names, dt_mesh.mesh_dim_names)

  @unittest.skipIf(xr.global_runtime_device_count() < 2,
                   "Multiple devices needed for 2D mesh test")
  def test_convert_2d_device_mesh(self):
    device_count = xr.global_runtime_device_count()
    mesh_shape = (2, device_count // 2)

    dt_mesh = DeviceMesh("xla", torch.arange(device_count).reshape(mesh_shape))

    xla_mesh = convert_to_xla_mesh(dt_mesh)

    self.assertIsInstance(xla_mesh, Mesh)
    self.assertEqual(len(xla_mesh.device_ids), device_count)
    self.assertEqual(xla_mesh.mesh_shape, mesh_shape)
    self.assertEqual(xla_mesh.axis_names, dt_mesh.mesh_dim_names)

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed for custom dim names test")
  def test_convert_with_custom_dim_names(self):
    device_count = xr.global_runtime_device_count()
    dt_mesh = DeviceMesh(
        "xla", list(range(device_count)), mesh_dim_names=["data_parallel"])

    xla_mesh = convert_to_xla_mesh(dt_mesh)

    self.assertIsInstance(xla_mesh, Mesh)
    self.assertEqual(len(xla_mesh.device_ids), device_count)
    self.assertEqual(xla_mesh.mesh_shape, (device_count,))
    self.assertEqual(xla_mesh.axis_names, ("data_parallel",))

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed for device IDs order test")
  def test_convert_mesh_device_ids_order(self):
    device_count = xr.global_runtime_device_count()
    device_ids = list(range(device_count))

    mesh_shape = (2, device_count // 2)
    mesh_2d = torch.tensor(device_ids).reshape(mesh_shape)
    dt_mesh = DeviceMesh("xla", mesh_2d)

    xla_mesh = convert_to_xla_mesh(dt_mesh)

    expected_flattened = mesh_2d.flatten().tolist()
    self.assertEqual(list(xla_mesh.device_ids), expected_flattened)

  @unittest.skipIf(xr.global_runtime_device_count() == 1,
                   "Multiple devices needed for mismatch test")
  def test_device_count_mismatch_assertion(self):
    device_count = xr.global_runtime_device_count()
    with self.assertRaises(AssertionError):
      dt_mesh = DeviceMesh("xla", list(range(device_count - 1)))
      convert_to_xla_mesh(dt_mesh)

  @unittest.skipIf(xr.global_runtime_device_count() < 4,
                   "At least 4 devices needed for mesh configuration tests")
  def test_mesh_configurations(self):
    device_count = xr.global_runtime_device_count()
    test_configs = [((1, device_count), "flat"),
                    ((2, device_count // 2), "2d_balanced")]

    for mesh_shape, config_name in test_configs:
      with self.subTest(configuration=config_name):
        dt_mesh = DeviceMesh("xla",
                             torch.arange(device_count).reshape(mesh_shape))
        xla_mesh = convert_to_xla_mesh(dt_mesh)

        self.assertEqual(xla_mesh.mesh_shape, mesh_shape)
        self.assertEqual(len(xla_mesh.device_ids), device_count)
        self.assertEqual(list(xla_mesh.device_ids), list(range(device_count)))

  def test_mesh_property_consistency(self):
    device_count = xr.global_runtime_device_count()
    dt_mesh = init_device_mesh("xla", mesh_shape=(device_count,))

    xla_mesh = convert_to_xla_mesh(dt_mesh)

    self.assertEqual(dt_mesh.size(), len(xla_mesh.device_ids))
    self.assertEqual(tuple(dt_mesh.mesh.size()), xla_mesh.mesh_shape)

    expected_device_ids = dt_mesh.mesh.flatten().tolist()
    self.assertEqual(list(xla_mesh.device_ids), expected_device_ids)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
