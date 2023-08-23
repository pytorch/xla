import copy

import unittest
from unittest.mock import patch
import os
import sys

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs

import test_xla_sharding_base


class BasicShardingTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  @patch.dict(os.environ, {"XLA_DUMP_POST_OPTIMIZATIONS": "1"})
  def test_xla_sharded_hlo_dump_post_optimizations(self):
    t1 = torch.randn(1, 128).to(xm.xla_device())
    t2 = torch.randn(128, 1).to(xm.xla_device())
    xs.mark_sharding(t1, self._get_mesh((1, self.n_devices)), (0, 1))

    t3 = t1 @ t2
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([t3])
    if self.n_devices > 1:
      self.assertIn('all-reduce', hlo)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
