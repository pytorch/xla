import unittest
import os
import sys
import subprocess

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.distributed.xla_backend
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
from torch_xla.amp import autocast
import torch_xla.debug.metrics as met

import test_xla_sharding_base


class BasicXMAPITest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_get_xla_supported_devices(self):
    device_type = os.environ['PJRT_DEVICE']
    devices = xm.get_xla_supported_devices(device_type)
    self.assertEqual(len(devices), 1)

  def test_world_size(self):
    self.assertEqual(xr.world_size(), 1)

  def test_get_ordinal(self):
    self.assertEqual(xr.global_ordinal(), 0)

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
    elif device_type == 'CUDA':
      command = 'nvidia-smi --list-gpus | wc -l'
      result = subprocess.run(
          command,
          capture_output=True,
          shell=True,
          check=True,
          text=True,
      )
      expected_gpu_cnt = int(result.stdout)
      self.assertEqual(xr.global_runtime_device_count(), expected_gpu_cnt)

  def test_addressable_runtime_device_count(self):
    device_type = os.environ['PJRT_DEVICE']
    if device_type == "TPU":
      self.assertGreaterEqual(xr.addressable_runtime_device_count(), 4)
    elif device_type == "CPU":
      self.assertEqual(xr.addressable_runtime_device_count(), 1)

  def test_runtime_spmd_api(self):
    met.clear_counters()
    self.assertTrue(xr.is_spmd())
    del os.environ["XLA_USE_SPMD"]
    self.assertFalse(xr.is_spmd())

    # unittest process can persist XLA_USE_SPMD from other test suites,
    # so t may be on a SPMD or non-SPMD device. If this test is run independently
    # outside unittest, then it lives on a non-SPMD device.
    t = torch.ones(2, 2).to(xm.xla_device())

    # Should enable SPMD without crashing.
    xr.use_spmd()
    self.assertTrue(xr.is_spmd())
    # TODO(yeounoh) check device type once tensor device becomes mutable

    # execute replicated
    t += 1
    torch_xla.sync(wait=True)
    self.assertEqual(met.counter_value("ExecuteReplicated"), 1)


class BasicAutocastAPITest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  @unittest.skipIf(xr.device_type() not in ['TPU', 'CUDA'],
                   f"TPU/GPU autocast test.")
  def test_xla_autocast_api(self):
    device = xm.xla_device()
    t1 = torch.ones([2, 3], device=device, dtype=torch.float32)
    t2 = torch.ones([3, 2], device=device, dtype=torch.float32)
    with autocast(device, dtype=torch.bfloat16):
      t3 = torch.matmul(t1, t2)
    expected_dtype = torch.bfloat16 if xr.is_bf16_supported() else torch.float16
    self.assertTrue(t3.dtype == expected_dtype)


class BasicDistributedTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    return super().setUpClass()

  def test_xla_backend(self):
    # XLA backend is not supported with SPMD
    with self.assertRaises(AssertionError):
      dist.init_process_group('xla', init_method='xla://')


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
