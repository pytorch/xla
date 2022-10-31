import os
import sys
import copy

import unittest
import numpy as np

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor


@unittest.skipIf((os.getenv('PJRT_DEVICE') == "") or (
    xm.get_xla_supported_devices("GPU") is not None
), "PyTorch/XLA SPMD requires PJRT_DEVICE={CPU, TPU}, GPU is currently not supported."
                )
class XlaShardingTest(unittest.TestCase):

  def test_xla_sharded_tensor(self):
    n_devices = xm.xrt_world_size()
    mesh_shape = (1, n_devices)
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xst1 = xs.mark_sharding(xt1, mesh_shape, partition_spec)

    # TODO(244003536) add more tests for XLAShardedTensror.
    self.assertTrue(isinstance(xst1, XLAShardedTensor))

  def test_mark_sharding_2d(self):
    t1 = torch.randn(1, 128, device='cpu')
    t2 = torch.randn(1, 128, device='cpu')
    expected = t1 @ t2.T

    xt1 = t1.to(xm.xla_device())
    xt2 = t2.to(xm.xla_device())
    n_devices = xm.xrt_world_size()
    xs.mark_sharding(xt1, (1, n_devices), (0, 1))
    annotation = '{devices=[1,%d]%s}' % (n_devices, ','.join(
        [str(i)
         for i in range(n_devices)])) if n_devices > 1 else '{maximal device=0}'
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt1))

    actual = (xt1 @ xt2.T).cpu()
    self.assertTrue(torch.allclose(expected, actual))

  def test_mark_sharding_4d(self):
    t = torch.randn(2, 4, 8, 16, device='cpu')
    expected = t + t

    xt = t.to(xm.xla_device())
    n_devices = xm.xrt_world_size()
    xs.mark_sharding(xt, (1, 1, 1, n_devices), (0, 1, 2, 3))
    annotation = '{devices=[1,1,1,%d]%s}' % (n_devices, ','.join(
        [str(i)
         for i in range(n_devices)])) if n_devices > 1 else '{maximal device=0}'
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt))

    actual = (xt + xt).cpu()
    self.assertTrue(torch.allclose(expected, actual))

  def test_clear_sharding(self):
    xt = torch.randn(2, 4, 8, 16).to(xm.xla_device())
    n_devices = xm.xrt_world_size()
    xs.mark_sharding(xt, (1, 1, 1, n_devices), (0, 1, 2, 3))
    self.assertTrue(torch_xla._XLAC._get_xla_sharding_spec(xt))
    xs.clear_sharding(xt)
    self.assertFalse(torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_deep_copy(self):
    xt = torch.randn(2, 4, 8, 16).to(xm.xla_device())
    n_devices = xm.xrt_world_size()
    xs.mark_sharding(xt, (1, 1, 1, n_devices), (0, 1, 2, 3))
    xt2 = copy.deepcopy(xt)
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(xt),
        torch_xla._XLAC._get_xla_sharding_spec(xt2))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
