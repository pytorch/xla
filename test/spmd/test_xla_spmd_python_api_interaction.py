import unittest
import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr

import test_xla_sharding_base


class BasicXMAPITest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def test_get_xla_supported_devices(self):
    device_type = os.environ['PJRT_DEVICE']
    devices = xm.get_xla_supported_devices(device_type)
    self.assertEqual(len(devices), 1)

  def test_world_size(self):
    self.assertEqual(xm.xrt_world_size(), 1)

  def test_get_ordinal(self):
    self.assertEqual(xm.get_ordinal(), 0)

  def test_get_local_ordinal(self):
    self.assertEqual(xm.get_local_ordinal(), 0)

  def test_is_master_ordinal(self):
    self.assertTrue(xm.is_master_ordinal())

  def test_xla_device(self):
    device = xm.xla_device()
    self.assertEqual(device, torch.device('xla:0'))

  def test_xla_real_devices(self):
    device = xm.xla_device()
    device_type = os.environ['PJRT_DEVICE']
    self.assertEqual(xm.xla_real_devices([device]), [device_type + ':0'])

  def test_xla_device_hw(self):
    device = xm.xla_device()
    device_type = os.environ['PJRT_DEVICE']
    replication_devices = xm.xla_replication_devices([device])
    self.assertEqual(xm.xla_device_hw(device), device_type)

  def test_xla_replication_devices(self):
    device = xm.xla_device()
    device_type = os.environ['PJRT_DEVICE']
    replication_devices = xm.xla_replication_devices([device])
    self.assertEqual(xm.xla_real_devices([device]), [device_type + ':0'])


class BasicRuntimeAPITest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def test_local_process_count(self):
    self.assertEqual(xr.local_process_count(), 1)

  def test_global_device_count(self):
    self.assertEqual(xr.global_device_count(), 1)

  def test_world_size(self):
    self.assertEqual(xr.world_size(), 1)

  def test_local_device_count(self):
    self.assertEqual(xr.local_device_count(), 1)

  def test_addressable_device_count(self):
    self.assertEqual(xr.addressable_device_count(), 1)

  def test_global_ordinal(self):
    self.assertEqual(xr.global_ordinal(), 0)

  def test_local_ordinal(self):
    self.assertEqual(xr.local_ordinal(), 0)

  def test_process_index(self):
    self.assertEqual(xr.process_index(), 0)

  def test_process_count(self):
    self.assertEqual(xr.process_count(), 1)

  def test_global_runtime_device_count(self):
    device_type = os.environ['PJRT_DEVICE']
    if device_type == "TPU":
      self.assertGreaterEqual(xr.global_runtime_device_count(), 4)
    elif device_type == "CPU":
      self.assertEqual(xr.global_runtime_device_count(), 1)

  def test_addressable_runtime_device_count(self):
    device_type = os.environ['PJRT_DEVICE']
    if device_type == "TPU":
      self.assertGreaterEqual(xr.addressable_runtime_device_count(), 4)
    elif device_type == "CPU":
      self.assertEqual(xr.addressable_runtime_device_count(), 1)

  def test_runtime_spmd_api(self):
    self.assertTrue(xr.is_spmd())
    del os.environ["XLA_USE_SPMD"]
    self.assertFalse(xr.is_spmd())
    # reset for other test cases
    os.environ["XLA_USE_SPMD"] = "1"


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
