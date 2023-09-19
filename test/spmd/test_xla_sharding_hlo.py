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
import torch_xla.debug.metrics as met

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

  def test_xla_replicated_input_output_alias(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(8, 8).to(xla_device)
    t2 = torch.randn(8, 8).to(xla_device)

    # check in place op aliasing.
    t3 = t1 + t2
    t1 *= 2.0
    t2 += 2.0
    xm.mark_step()
    # expected post-optimization HLO with "XLA_FLAGS=--xla_dump_to=/tmp/hlo_dir"
    # HloModule SyncTensorsGraph.18, is_scheduled=true, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }, entry_computation_layout={(f32[8,8]{1,0:T(8,128)}, f32[8,8]{1,0:T(8,128)}, f32[]{:T(128)})->(f32[8,8]{1,0:T(8,128)}, f32[8,8]{1,0:T(8,128)}, f32[8,8]{1,0:T(8,128)})}, allow_spmd_sharding_propagation_to_output={true}
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)

  def test_xla_sharded_input_output_alias(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(8, 8).to(xla_device)
    t2 = torch.randn(8, 8).to(xla_device)

    xs.mark_sharding(t1, self._get_mesh((1, self.n_devices)), (0, 1))
    xs.mark_sharding(t2, self._get_mesh((1, self.n_devices)), (0, 1))

    # check in place op aliasing.
    t3 = t1 + t2
    t1 *= 2.0
    t2 += 2.0
    xm.mark_step()
    # expected post-optimization HLO with "XLA_FLAGS=--xla_dump_to=/tmp/hlo_dir"
    # HloModule SyncTensorsGraph.18, is_scheduled=true, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }, entry_computation_layout={(f32[8,8]{1,0:T(8,128)}, f32[8,8]{1,0:T(8,128)}, f32[]{:T(128)})->(f32[8,8]{1,0:T(8,128)}, f32[8,8]{1,0:T(8,128)}, f32[8,8]{1,0:T(8,128)})}, allow_spmd_sharding_propagation_to_output={true}
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
