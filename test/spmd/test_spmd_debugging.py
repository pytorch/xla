import os
import sys
import numpy as np
import unittest

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh
import torch_xla.distributed.spmd
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding


# @unittest.skipIf(not using_pjrt() or xm.get_xla_supported_devices("GPU"),
#                  f"Requires PJRT_DEVICE set to `TPU` or `CPU`.")
# class XlaShardingTest(unittest.TestCase):

import test_xla_sharding_base

class DebuggingSpmdTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    os.environ["XLA_USE_SPMD"] = "1"
    super().setUpClass()

  def test_debugging_spmd_single_host_tiled(self):
    # xr.use_spmd()
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
    # xr.use_spmd()
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
    # xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

    partition_spec_replicated = (None, None)
    t = torch.randn(8, 32, device=device)
    # xs.mark_sharding(t, mesh, (0, None))
    xs.mark_sharding(t, mesh, partition_spec_replicated)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    print("sharding is: ")
    print(sharding)
    print("then print: ")
    visualize_tensor_sharding(t)



# test_single_host_tiled()
# test_single_host_partial_replication()
# test_single_host_replicated()

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
