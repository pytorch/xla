import os
import sys

import unittest

import torch
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.experimental.xla_sharding as xs
import test_xla_sharding_base


class VirtualDeviceTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    os.environ["XLA_USE_SPMD"] = "1"
    super().setUpClass()

  def test_mark_sharding(self):
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), partition_spec)
    self.assertTrue(
        torch.allclose(
            xt1 + 0,
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8],
                         dtype=torch.float,
                         device=xm.xla_device())))

  def test_metrics_recorded(self):
    met.clear_counters()
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), partition_spec)
    self.assertIn("VirtualDeviceUsage", met.counter_names())
    self.assertNotEqual(met.counter_value("VirtualDeviceUsage"), 0)

  def test_model_weight_metrics(self):
    met.clear_counters()
    partition_spec = (0, 1)
    model = nn.Linear(128, 64).to(xm.xla_device())
    xs.mark_sharding(model.weight, self._get_mesh((1, self.n_devices)),
                     partition_spec)
    self.assertNotIn("VirtualDeviceUsage", met.counter_names())

  def test_no_sharding(self):
    t1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                      dtype=torch.float,
                      device=xm.xla_device())
    t2 = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]],
                      dtype=torch.float,
                      device=xm.xla_device())
    t3 = t1 + t2
    t3_expected = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    self.assertEqual(t3.tolist()[0], t3_expected)

  def test_outbound_data_metrics(self):
    partition_spec = (0, 1)

    met.clear_all()
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), partition_spec)
    outbound_with_virtual_device = met.metric_data("OutboundData")[1]

    os.environ["XLA_USE_SPMD"] = "0"

    met.clear_all()
    xt2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt2, self._get_mesh((1, self.n_devices)), partition_spec)
    outbound_without_virtual_device = met.metric_data("OutboundData")[1]

    self.assertLess(outbound_with_virtual_device,
                    outbound_without_virtual_device)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
