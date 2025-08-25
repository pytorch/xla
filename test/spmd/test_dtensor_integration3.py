import os
import sys

import torch
from torch import nn
import torch.optim as optim
from torch.distributed.tensor import (DeviceMesh, Replicate, Shard,
                                      distribute_tensor, distribute_module,
                                      init_device_mesh)
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import auto_policy

import unittest

import test_xla_sharding_base


# This integration test passes when run independently.
class DTensorIntegrationTest3(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  # This test fails with functionalization, so disabled functionalization.
  def test_xla_placement(self):

    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(32, 16, bias=False)
        self.out_proj = torch.nn.Linear(16, 8, bias=False)

      def forward(self, hidden):
        hidden = self.in_proj(hidden)
        hidden = torch.relu(hidden)
        hidden = self.out_proj(hidden)
        return hidden

    def forward_pure(hidden, in_proj_weight, out_proj_weight):
      hidden = torch.matmul(hidden, in_proj_weight.T)
      hidden = torch.relu(hidden)
      hidden = torch.matmul(hidden, out_proj_weight.T)
      return hidden

    #xr.use_spmd()
    model = Model()
    model.to('xla')
    device_count = xr.global_runtime_device_count()
    device_mesh = init_device_mesh(
        device_type='xla', mesh_shape=(device_count,))

    # Tensor parallel shardings
    inputs_sharding = [Replicate()]
    in_proj_weight_sharding = [Shard(0)]
    out_proj_weight_sharding = [Shard(1)]

    torch.manual_seed(15213)
    inputs = torch.rand(2, 32)
    inputs = inputs.to('xla')
    outputs_unsharded = model(inputs)
    xm.mark_step()
    outputs_unsharded = outputs_unsharded.cpu()
    inputs = distribute_tensor(inputs, device_mesh, placements=inputs_sharding)
    in_proj_weight = distribute_tensor(
        model.in_proj.weight, device_mesh, placements=in_proj_weight_sharding)
    out_proj_weight = distribute_tensor(
        model.out_proj.weight, device_mesh, placements=out_proj_weight_sharding)
    outputs_sharded = forward_pure(inputs, in_proj_weight, out_proj_weight)
    xm.mark_step()
    outputs_sharded = outputs_sharded.cpu()
    #from torch_xla.distributed.spmd.debugging import visualize_sharding
    #generated_table = visualize_sharding(outputs.sharding_spec(), use_color=False)
    print(outputs_unsharded)
    print(outputs_sharded)
    torch.testing.assert_close(outputs_sharded.global_tensor.numpy(),
                               outputs_unsharded.detach().numpy())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
