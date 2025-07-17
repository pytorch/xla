import sys
import unittest
import torch
import torch_xla
from torch_xla.core.xla_builder import create_placeholder_tensor
import torch_xla.debug.metrics as met
import re
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

import test_xla_sharding_base


class TestSPMDPlaceholder(test_xla_sharding_base.XlaShardingTest):

  def setUp(self):
    super().setUpClass()

  def test_create_placeholder(self):
    num_devices = self.n_devices
    for shape, dtype in zip(
        ((num_devices, num_devices), (num_devices, num_devices, 2),
         (num_devices, num_devices, 2, 2)),
        (torch.float32, torch.bfloat16, torch.int8),
    ):
      model_axis = max(1, self.n_devices // 2)
      data_axis = self.n_devices // model_axis
      mesh_shape = (data_axis, model_axis) + (1,) * (len(shape) - 2)
      axis_names = ('x', 'y') + tuple(f'z{i}' for i in range(1, len(shape) - 1))
      mesh = self._get_mesh(mesh_shape, axis_names=axis_names)

      p = create_placeholder_tensor(shape, dtype)
      xs.mark_sharding(p, mesh, axis_names)
      assert isinstance(p, torch.Tensor)
      assert p.device == torch_xla.device()
      self.assertEqual(p.dtype, dtype)
      self.assertEqual(p.shape, shape)
      self.assertTrue(torch_xla._XLAC._is_placeholder(p))

  def test_read_value_crashes(self):
    mesh = self._get_mesh((self.n_devices,), axis_names=('x',))
    p = create_placeholder_tensor((self.n_devices,), torch.bfloat16)
    xs.mark_sharding(p, mesh, ('x',))
    with self.assertRaises(RuntimeError):
      p.cpu()

  def test_trace_graph(self):
    met.clear_all()
    self.assertFalse(met.metric_data("TransferToDeviceTime"))

    model_axis = max(1, self.n_devices // 2)
    data_axis = self.n_devices // model_axis
    mesh_shape = (data_axis, model_axis)
    mesh = self._get_mesh(mesh_shape, axis_names=('x', 'y'))

    p1 = create_placeholder_tensor((128, 32), torch.bfloat16)
    xs.mark_sharding(p1, mesh, ('x', 'y'))
    a = torch.sin(p1)

    p2 = create_placeholder_tensor((32, 64), torch.bfloat16)
    xs.mark_sharding(p2, mesh, ('x', 'y'))
    # We use p1 once and p2 twice. But the graph should still only have two parameters.
    b = (a @ p2) @ p2.T
    ir: str = torch_xla._XLAC._get_xla_tensors_text([b])
    self.assertEqual(ir.count("xla::device_data()"), 2)
    self.assertEqual(ir.count("bf16[32,64]{1,0} xla::device_data()"), 1)
    self.assertEqual(ir.count("bf16[128,32]{1,0} xla::device_data()"), 1)
    hlo: str = torch_xla._XLAC._get_xla_tensors_hlo([b])
    regex = r'\(p.*: bf16\[32,64\], p.*: bf16\[128,32\]\) -> \(bf16\[128,32\]\)'
    assert re.search(regex, hlo) is not None

    # There should be no buffers transferred to the device during tracing
    self.assertFalse(met.metric_data("TransferToDeviceTime"))

  def test_placeholder_handle_unique(self):
    mesh = self._get_mesh((self.n_devices,), axis_names=('x',))

    p1 = create_placeholder_tensor((self.n_devices,), torch.bfloat16)
    xs.mark_sharding(p1, mesh, ('x',))

    p2 = create_placeholder_tensor((self.n_devices,), torch.bfloat16)
    xs.mark_sharding(p2, mesh, ('x',))

    h1, h2 = torch_xla._XLAC._get_tensors_handle([p1, p2])
    self.assertNotEqual(h1, h2)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
