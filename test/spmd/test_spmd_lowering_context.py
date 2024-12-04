import sys

import unittest

import test_xla_sharding_base

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
import contextlib


class TestSPMDLoweringContext(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_device_parameter_id_tensor_mapping(self):
    met.clear_all()

    model_axis = min(8, self.n_devices)
    data_axis = self.n_devices // model_axis
    mesh_shape = (data_axis, model_axis)
    spmd_mesh = self._get_mesh(mesh_shape, axis_names=('x', 'y'))

    device = xm.xla_device()
    a = torch.randn([32, 2048]).to(device)
    xs.mark_sharding(a, spmd_mesh, ('x', 'y'))
    b = torch.ones(2048).to(device)
    xs.mark_sharding(b, spmd_mesh, ('x',))

    def fn(a, b):
      return a + b

    result = fn(a, b)
    ctx = torch_xla._XLAC.lowering.LoweringContext("MyCustomName")
    ctx.build([result])
    torch_xla.sync()

    mapping = ctx.device_parameter_id_tensor_mapping()
    num_params = len(mapping)
    self.assertEqual(num_params, 2)
    self.assertNotEqual(ctx.tensor_parameter_id(a), -1)
    self.assertNotEqual(ctx.tensor_parameter_id(b), -1)
    self.assertEqual(met.counter_value("VirtualDeviceUsage"), num_params)

    # Ensure that the parameter mapping does not require transferring data
    # from the device to the host when sharded.
    self.assertFalse(met.metric_data("TransferFromDeviceTime"))
    self.assertFalse(met.counter_value("ReplicateShardedData"))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
