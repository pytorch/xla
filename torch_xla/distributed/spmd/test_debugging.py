import torch
import torch_xla
from torch_xla.distributed.fsdp.debugging import visualize_tensor_sharding

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh


def test_single_host_tiled():
  xr.use_spmd()
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (2, num_devices // 2)
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
  t = torch.randn(8, 4).to(xm.xla_device())
  partition_spec = (0, 1)
  m1_sharded = xs.mark_sharding(t, mesh, partition_spec)
  sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
  print("sharding is:")
  print(sharding)
  print("then print:")
  visualize_tensor_sharding(t)

def test_single_host_partial_replication():
  xr.use_spmd()
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (2, num_devices // 2)
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

  partition_spec = (0, None)
  t = torch.randn(8, 32).to(xm.xla_device())
  xs.mark_sharding(t, mesh, (0, None))
  sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
  print("sharding is: ")
  print(sharding)
  print("then print: ")
  visualize_tensor_sharding(t)

def test_single_host_replicated():
  xr.use_spmd()
  num_devices = xr.global_runtime_device_count()
  mesh_shape = (2, num_devices // 2)
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

  partition_spec_replicated = (None, None)
  t = torch.randn(8, 32).to(xm.xla_device())
  # xs.mark_sharding(t, mesh, (0, None))
  sharded = xs.mark_sharding(t, mesh, partition_spec_replicated)
  sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
  print("sharding is: ")
  print(sharding)
  print("then print: ")
  visualize_tensor_sharding(t)

test_single_host_tiled()
test_single_host_partial_replication()
test_single_host_replicated()
