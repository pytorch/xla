import os
import sys

import unittest

import torch
from torch import nn
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import test_xla_sharding_base


class VirtualDeviceTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
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
    self.assertIn("VirtualDeviceUsage", met.counter_names())
    self.assertNotEqual(met.counter_value("VirtualDeviceUsage"), 0)

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

  def test_no_sharding_1d(self):
    t1 = torch.arange(9, dtype=torch.float, device=xm.xla_device())
    t2 = torch.arange(9, dtype=torch.float, device=xm.xla_device())
    t3 = t1 + t2
    t3_expected = list(range(0, 18, 2))
    self.assertEqual(t3.tolist(), t3_expected)

  def test_outbound_data_metrics(self):
    partition_spec = (0, 1)

    met.clear_all()
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), partition_spec)
    outbound_with_virtual_device = met.metric_data("OutboundData")[1]

    # Without virtual device optimization, we expect the data to be transferred to
    # device at least twice, so assert that the actual transfer amount is less.
    self.assertLess(outbound_with_virtual_device,
                    2 * xt1.nelement() * xt1.element_size())

  def test_non_tensor_scalar(self):
    sharding_spec = xs.ShardingSpec(self._get_mesh((1, self.n_devices)), (0, 1))
    # tensor will have device as `SPMD:0` in c++
    xt1 = xm.send_cpu_data_to_device([torch.randn(3, 3)],
                                     xm.xla_device(),
                                     input_sharding=sharding_spec)[0]
    # we will transfer 0.5 as a device_data to the 'SPMD:0' device, need to make sure
    # that virtual device can handle this case.
    xt2 = xt1 / 0.5
    torch_xla.sync(wait=True)
    torch.allclose(xt2.cpu(), xt1.cpu() / 0.5)

  def test_sync_on_virtual_device(self):
    torch_xla.sync()
    sharding_spec = xs.ShardingSpec(self._get_mesh((1, self.n_devices)), (0, 1))
    # tensor will have device as `SPMD:0` in c++
    xt1 = xm.send_cpu_data_to_device([torch.randn(3, 3)],
                                     xm.xla_device(),
                                     input_sharding=sharding_spec)[0]
    xt2 = xt1 / 0.5
    torch_xla.sync(wait=True)
    # after `torch_xla.sync()`, xt2 should be materalized
    self.assertNotIn('aten::div',
                     torch_xla._XLAC._get_xla_tensor_debug_info(xt2))

  def test_virtual_device_no_upload(self):
    met.clear_all()
    device = xm.xla_device()
    t1 = torch.randn(5, 5).to(device)
    t1_debug_info = torch_xla._XLAC._get_xla_tensor_debug_info(t1)
    # t1's upload to device should be deferred
    self.assertIn("Tensor on host: with size [5, 5]", t1_debug_info)
    self.assertNotIn("TransferToDeviceTime", met.metric_names())
    # t1 should be on SPMD device under spmd context
    self.assertIn("Device: SPMD:0", t1_debug_info)
    self.assertIn("IR: None", t1_debug_info)
    self.assertIn("XLAData: None", t1_debug_info)

  def test_virtual_device_upload_after_mark_sharding(self):
    met.clear_all()
    partition_spec = (0, 1)
    device = xm.xla_device()
    t1 = torch.randn(8, 8).to(device)
    t1_debug_info = torch_xla._XLAC._get_xla_tensor_debug_info(t1)
    self.assertIn("Tensor on host: with size [8, 8]", t1_debug_info)
    xs.mark_sharding(t1, self._get_mesh((1, self.n_devices)), partition_spec)
    t1_debug_info_new = torch_xla._XLAC._get_xla_tensor_debug_info(t1)
    # tensor should be uploaded to device after mark_sharding
    self.assertIn("Tensor on host: None", t1_debug_info_new)
    self.assertIn("xla::device_data", t1_debug_info_new)
    self.assertIn("XLAShardedData", t1_debug_info_new)
    self.assertIn("TransferToDeviceTime", met.metric_names())

  def test_virtual_device_upload_after_tracing(self):
    met.clear_all()
    device = xm.xla_device()
    t1 = torch.randn(8, 8).to(device)
    t1_debug_info = torch_xla._XLAC._get_xla_tensor_debug_info(t1)
    self.assertIn("Tensor on host: with size [8, 8]", t1_debug_info)
    t2 = t1 + t1
    t1_debug_info_new = torch_xla._XLAC._get_xla_tensor_debug_info(t1)
    # tensor should be uploaded to device after being used as input to other op.
    self.assertIn("Tensor on host: None", t1_debug_info_new)
    self.assertIn("xla::device_data", t1_debug_info_new)
    self.assertIn("TransferToDeviceTime", met.metric_names())

  def test_virtual_device_upload_for_sharded_dataloader(self):
    met.clear_counters()
    device = xm.xla_device()
    sharding_spec = xs.ShardingSpec(self._get_mesh((1, self.n_devices)), (0, 1))
    # tensor will have device as `SPMD:0` in c++
    t1 = xm.send_cpu_data_to_device([torch.randn(8, 8)],
                                    device,
                                    input_sharding=sharding_spec)[0]
    t1_debug_info = torch_xla._XLAC._get_xla_tensor_debug_info(t1)
    self.assertIn("Device: SPMD:0", t1_debug_info)
    # tensor should be uploaded to device after send_cpu_data_to_device + sharding_spec
    self.assertIn("Tensor on host: None", t1_debug_info)
    self.assertIn("xla::device_data", t1_debug_info)
    self.assertIn("XLAShardedData", t1_debug_info)
    self.assertIn("TransferToDeviceTime", met.metric_names())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
