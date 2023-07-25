import unittest
import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import test_xla_sharding_base


class BasicXMAPITest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    os.environ["XLA_USE_SPMD"] = "1"
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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
