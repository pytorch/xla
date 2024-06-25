import unittest
import sys

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import numpy as np


class Eager(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    torch_xla.experimental.eager_mode(True)
    xr.use_spmd()
    cls.n_devices = xr.global_runtime_device_count()
    cls.device_ids = np.array(range(cls.n_devices))

  def _get_mesh(self, mesh_shape, device_ids=None, axis_names=None):
    assert type(mesh_shape) is tuple, 'mesh_shape must be Tuple[int]'
    if device_ids is None:
      device_ids = self.device_ids
    assert len(device_ids) == self.n_devices
    return xs.Mesh(device_ids, mesh_shape, axis_names)

  def test_eager_spmd_basic(self):
    device = torch_xla.device()
    mesh = self._get_mesh((self.n_devices,), axis_names=('data',))
    torch.manual_seed(100)
    linear = torch.nn.Linear(10, 20)
    input = torch.randn(8, 10)
    input_xla = input.to(device)
    res = linear(input)
    linear.to(device)
    res_xla = linear(input_xla)
    self.assertTrue(torch.allclose(res, res_xla.cpu(), rtol=1e-3))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
