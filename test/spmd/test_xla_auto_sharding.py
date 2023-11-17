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
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import XLAShardedTensor
import test_xla_sharding_base

import torch_xla.core.xla_env_vars as xenv
import torch_xla.utils.utils as xu
from torch_xla._internal import tpu


class BasicShardingTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd(auto=True)
    super().setUpClass()

  def test_xla_sharded_tensor(self):
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xst1 = xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)),
                            partition_spec)

    # TODO(244003536) add more tests for XLAShardedTensror.
    self.assertTrue(isinstance(xst1, XLAShardedTensor))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
