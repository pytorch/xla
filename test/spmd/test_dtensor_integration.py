import os
import sys

import torch
from torch import nn
import torch.optim as optim
from torch.distributed._tensor import DeviceMesh, Shard
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import xla_distribute_tensor

import unittest

import test_xla_sharding_base


class DTensorIntegrationTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def test_xla_distribute_tensor(self):
    device_count = xr.global_runtime_device_count()
    device_mesh = DeviceMesh("xla", list(range(device_count)))
    shard_spec = [Shard(0)]

    for requires_grad in [True, False]:
      tensor_to_shard = torch.randn(
          3 * device_count,
          3,
          requires_grad=requires_grad,
          device=xm.xla_device())
      dist_tensor = xla_distribute_tensor(tensor_to_shard, device_mesh,
                                          shard_spec)
      # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
      assert type(dist_tensor).__name__ == "XLAShardedTensor"
      assert len(dist_tensor.sharding_spec) > 0

      global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
      self.assertEqual(global_tensor.size(), torch.Size([3 * device_count, 3]))
      local_tensor = dist_tensor.local_shards[0].data
      self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
      if requires_grad:
        self.assertTrue(dist_tensor.global_tensor.requires_grad)
        self.assertTrue(dist_tensor.is_leaf)

  def test_optimizer_step_with_sharding(self):
    # Use simple linear model to test model parameter sharding
    model = self.SimpleLinear().to(xm.xla_device())

    # Running the same mark_sharding test with xla_distribute_tensor instead
    device_count = xr.global_runtime_device_count()
    device_mesh = DeviceMesh("xla", list(range(device_count)))
    shard_spec = [Shard(0)]
    xla_distribute_tensor(model.fc1.weight, device_mesh, shard_spec)
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    data = torch.randn(128, 128).to(xm.xla_device())
    target = torch.zeros(128).to(xm.xla_device())
    loss_fn = nn.CrossEntropyLoss()
    for i in range(3):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      xm.mark_step()
      # Sharding is persisted across mark_step calls, and test if the sharded computation
      # can repeat more than once without crashing.
      self.assertEqual(sharding_spec,
                       torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
