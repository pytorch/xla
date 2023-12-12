import unittest
import os
import sys

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

import test_xla_sharding_base
import torch_xla.experimental.spmd_fully_sharded_data_parallel as fsdp


class BasicShardingTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def _print_children_name(self, m):
    for name, child in m.named_children():
      print(name)
      self._print_children_name(child)

  def test_fsdp_v2(self):
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    model.fc1 = fsdp.SpmdFullyShardedDataParallel(model.fc1, mesh)
    model.fc2 = fsdp.SpmdFullyShardedDataParallel(model.fc2, mesh)
    model = fsdp.SpmdFullyShardedDataParallel(model, mesh)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(16, 128).to(xm.xla_device())
    xs.mark_sharding(x, mesh, ('fsdp', None))

    optimizer.zero_grad()
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    xm.mark_step()


    self._print_children_name(model)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
