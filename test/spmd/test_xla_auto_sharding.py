import copy

import unittest
from unittest.mock import patch
import math
import numpy as np
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import XLAShardedTensor
import test_xla_sharding_base

import torch_xla.core.xla_env_vars as xenv
import torch_xla.utils.utils as xu
from torch_xla._internal import tpu


class XlaAutoShardingTest(test_xla_sharding_base.XlaShardingTest):

  # initialized in setUpClass
  hash_no_auto = ""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    assert not torch_xla._XLAC._xla_get_auto_sharding()
    cls.init_test_variables(cls)
    assert cls.hash_no_auto
    xr.use_spmd(auto=True)

  def init_test_variables(cls):
    xt_no_auto = torch.ones(2, 2).to(xm.xla_device())
    cls.hash_no_auto = torch_xla._XLAC._get_graph_hash([xt_no_auto + 0])

  def test_auto_sharding_hashing(self):
    xt = torch.ones(2, 2).to(xm.xla_device())
    assert torch_xla._XLAC._xla_get_auto_sharding()
    hash_auto_spmd = torch_xla._XLAC._get_graph_hash([xt + 0])
    self.assertNotEqual(hash_auto_spmd, self.hash_no_auto)
    os.environ['XLA_AUTO_SPMD_MESH'] = '2,2'
    hash_auto_spmd2 = torch_xla._XLAC._get_graph_hash([xt + 0])
    self.assertNotEqual(hash_auto_spmd, hash_auto_spmd2)
    del os.environ['XLA_AUTO_SPMD_MESH']

  @unittest.skipUnless(xr.device_type() in ["TPU", "CPU"],
                       "Auto-sharding currently supports TPU & CPU backends.")
  def test_matmul(self):
    met.clear_counters()
    t1 = torch.ones(64, 128)
    t2 = torch.ones(128, 256)
    t3 = (t1 @ t2).sum()

    xt1 = t1.to(xm.xla_device())
    xt2 = t2.to(xm.xla_device())
    xt3 = (xt1 @ xt2).sum()
    torch_xla.sync()
    self.assertEqual(met.counter_value("CompileWithAutoSharding"), 1)
    self.assertTrue(torch.allclose(t3, xt3.cpu()))

  @unittest.skipUnless(xr.device_type() in ["TPU", "CPU"],
                       "Auto-sharding currently supports TPU & CPU backends.")
  def test_simple_linear_training(self):
    met.clear_counters()

    model = self.SimpleLinear().to(xm.xla_device())
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    data = torch.randn(128, 128).to(xm.xla_device())
    target = torch.zeros(128).to(xm.xla_device())
    loss_fn = nn.CrossEntropyLoss()
    for i in range(5):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      torch_xla.sync()
    cnt = met.counter_value("CompileWithAutoSharding")
    self.assertTrue((cnt is not None) and (cnt <= 3))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
