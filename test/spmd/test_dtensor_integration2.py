import os
import sys

import torch
from torch import nn
import torch.optim as optim
from torch.distributed.tensor import (DeviceMesh, Shard, distribute_tensor,
                                      distribute_module)
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import auto_policy

import unittest

import test_xla_sharding_base


# This integration test passes when run independently.
class DTensorIntegrationTest2(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  @unittest.skipUnless(xr.device_type() in ["TPU", "CPU"],
                       "Auto-sharding currently supports TPU device.")
  def test_xla_distribute_module_auto(self):
    device_count = xr.global_runtime_device_count()
    device_mesh = DeviceMesh("xla", list(range(device_count)))

    # Use torch_xla.distributed.spmd.auto_policy to enable auto-sharding;
    # Currently, model should be loaded to xla device via distribute_module.
    model = self.SimpleLinear()
    sharded_model = distribute_module(model, device_mesh, auto_policy)
    sharded_model.train()
    self.assertTrue(torch_xla._XLAC._xla_get_auto_sharding())

    optimizer = optim.SGD(sharded_model.parameters(), lr=0.1)
    data = torch.randn(128, 128).to(xm.xla_device())
    target = torch.zeros(128).to(xm.xla_device())
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(5):
      optimizer.zero_grad()
      output = sharded_model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      torch_xla.sync()
    # Should compile with auto-sharding, we expect up to 3 times
    cnt = met.counter_value("CompileWithAutoSharding")
    self.assertTrue((cnt is not None) and (cnt <= 3))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
