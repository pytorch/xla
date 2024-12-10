import sys

import unittest
from unittest.mock import patch
import math
import numpy as np
import os

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import test_xla_sharding_base


class SpmdGraphDumpTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_dump_with_output_sharding(self):
    save_file = os.getenv('XLA_SAVE_TENSORS_FILE')
    save_format = os.getenv('XLA_SAVE_TENSORS_FMT')
    assert save_file, "This test should be run with XLA_SAVE_TENSORS_FILE"
    should_dump_output_sharding = (save_format == 'hlo')
    save_file += '.0'
    device = xm.xla_device()
    xla_x = torch.randn(8, 32).to(device)
    xla_y = torch.randn(8, 32).to(device)
    # shard one of the input tensor
    partition_spec = (0, 1)
    xla_sharded_x = xs.mark_sharding(xla_x, self._get_mesh((1, self.n_devices)),
                                     partition_spec)
    xla_res = xla_x + xla_y
    xm.mark_step()
    with open(save_file, 'rb') as f:
      lines = f.readlines()
    self.assertGreater(len(lines), 0)
    if should_dump_output_sharding:
      self.assertIn('OUTPUT_SHARDING_END', str(lines[-2]))
    else:
      self.assertIn('END_GRAPH', str(lines[-2]))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
