import os
import unittest
import numpy as np

from torch import nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import torch_xla.core.xla_env_vars as xenv
import torch_xla.utils.utils as xu


class XlaShardingTest(unittest.TestCase):

  class SimpleLinear(nn.Module):

    def __init__(self):
      super(XlaShardingTest.SimpleLinear, self).__init__()
      self.fc1 = nn.Linear(128, 64)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
      y = self.relu(self.fc1(x))
      z = self.fc2(y)
      return z

  n_devices = 0
  device_ids = None

  @classmethod
  def setUpClass(cls):
    cls.n_devices = xr.global_runtime_device_count()
    cls.device_ids = np.array(range(cls.n_devices))
    xr.use_spmd()

  @classmethod
  def tearDownClass(cls):
    del os.environ['XLA_USE_SPMD']
    del os.environ['XLA_AUTO_SPMD']

  def _get_mesh(self, mesh_shape, device_ids=None, axis_names=None):
    assert type(mesh_shape) is tuple, 'mesh_shape must be Tuple[int]'
    if device_ids is None:
      device_ids = self.device_ids
    assert len(device_ids) == self.n_devices
    return xs.Mesh(device_ids, mesh_shape, axis_names)

  def _get_hybrid_mesh(self, ici_mesh_shape, axis_names=None):
    return xs.HybridMesh(ici_mesh_shape=ici_mesh_shape, axis_names=axis_names)
