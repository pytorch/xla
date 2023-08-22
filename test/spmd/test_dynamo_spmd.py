import os
import sys

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.debug.metrics as met
import unittest

import test_xla_sharding_base


class SimpleLinear(nn.Module):

  def __init__(self):
    super(SimpleLinear, self).__init__()
    self.fc1 = nn.Linear(128, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 1)
    # Add an additional 1x1 layer at the end to ensure the final layer
    # is not sharded.
    self.fc3 = nn.Linear(1, 1)

  def forward(self, x):
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return self.fc3(z)


class DynamoSpmdInferenceTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def test_dynamo_spmd_basic(self):
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    xs.mark_sharding(linear.fc2.weight, self._get_mesh((1, self.n_devices)),
                     (1, 0))
    xla_res = linear(xla_x)
    xm.mark_step()

    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    torch.allclose(xla_res.cpu(), dynamo_res.cpu())
    # TODO(JackCaoG): add counter checks after ExecuteReplicated also creates
    # a ExecuteMetric.

  @unittest.skip(
      "test is flaky, UncachedOutputSharding sometime doesn't show up. most likely a waitdeviceop issue"
  )
  def test_dynamo_spmd_output_sharding_cache(self):
    met.clear_all()
    device = xm.xla_device()
    linear = SimpleLinear().to(device)
    linear.eval()
    xla_x = torch.randn(1, 128, device=device)
    xla_y = torch.randn(1, 128, device=device)
    xs.mark_sharding(linear.fc2.weight, self._get_mesh((1, self.n_devices)),
                     (1, 0))
    dynamo_linear = torch.compile(linear, backend="openxla")
    dynamo_res = dynamo_linear(xla_x)
    xm.wait_device_ops()
    self.assertIn('UncachedOutputSharding', met.counter_names())
    self.assertEqual(met.counter_value('UncachedOutputSharding'), 1)
    dynamo_res = dynamo_linear(xla_y)
    self.assertEqual(met.counter_value('UncachedOutputSharding'), 1)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
