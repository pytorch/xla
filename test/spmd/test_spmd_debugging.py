import sys

import unittest
from unittest.mock import patch
import math
import numpy as np
import os

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding


import test_xla_sharding_base

@unittest.skipIf(not using_pjrt() or xm.get_xla_supported_devices("GPU")
                 or xm.get_xla_supported_devices("CPU"),
                 f"Requires PJRT_DEVICE set to `TPU`.")
class DebuggingSpmdTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()# os.environ["XLA_USE_SPMD"] = "1"
    super().setUpClass()

  def test_debugging_spmd_single_host_tiled(self):
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
    t = torch.randn(8, 4, device=device)
    partition_spec = (0, 1)
    xs.mark_sharding(t, mesh, partition_spec)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    print("sharding is:")
    print(sharding)
    print("then print:")
    visualize_tensor_sharding(t)


  def test_single_host_partial_replication():
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

    partition_spec = (0, None)
    t = torch.randn(8, 32,  device=device)
    xs.mark_sharding(t, mesh, (0, None))
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    print("sharding is: ")
    print(sharding)
    print("then print: ")
    visualize_tensor_sharding(t)


  def test_single_host_replicated():
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

    partition_spec_replicated = (None, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, partition_spec_replicated)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    print("sharding is: ")
    print(sharding)
    print("then print: ")
    visualize_tensor_sharding(t)

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
