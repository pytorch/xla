import os
import sys
import copy

import unittest
import numpy as np

import torch
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.pjrt import using_pjrt


@unittest.skipIf(not using_pjrt() or xm.get_xla_supported_devices("GPU"),
                 f"Requires PJRT_DEVICE set to `TPU` or `CPU`.")
class XlaShardingTest(unittest.TestCase):

  n_devices = 0
  device_ids = None

  @classmethod
  def setUpClass(cls):
    cls.n_devices = len(xm.get_xla_supported_devices())
    cls.device_ids = np.array(range(cls.n_devices))

  def _get_mesh(self, mesh_shape, device_ids=None):
    if device_ids is None:
      device_ids = self.device_ids
    assert len(device_ids) == self.n_devices
    return xs.Mesh(device_ids, mesh_shape)

  def test_xla_sharded_tensor(self):
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xst1 = xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)),
                            partition_spec)

    # TODO(244003536) add more tests for XLAShardedTensror.
    self.assertTrue(isinstance(xst1, XLAShardedTensor))

  def test_custom_tile_assignment(self):
    xt = torch.randn(10, 20).to(device=xm.xla_device())
    mesh_shape = (1, self.n_devices)
    device_ids = np.flip(self.device_ids)
    mesh = self._get_mesh(mesh_shape, device_ids)
    xs.mark_sharding(xt, mesh, (0, 1))
    annotation = '{devices=[1,%d]%s}' % (self.n_devices, ','.join([
        str(i) for i in reversed(range(self.n_devices))
    ])) if self.n_devices > 1 else '{maximal device=0}'
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_mark_sharding_2d(self):
    t1 = torch.randn(1, 128, device='cpu')
    t2 = torch.randn(1, 128, device='cpu')
    expected = t1 + t2

    xt1 = t1.to(xm.xla_device())
    xt2 = t2.to(xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), (0, 1))
    annotation = '{devices=[1,%d]%s}' % (self.n_devices, ','.join([
        str(i) for i in range(self.n_devices)
    ])) if self.n_devices > 1 else '{maximal device=0}'
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt1))

    actual = (xt1 + xt2).cpu()
    self.assertTrue(torch.allclose(expected, actual))

  def test_mark_sharding_4d(self):
    t = torch.randn(2, 4, 8, 16, device='cpu')
    expected = t + t

    xt = t.to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, 1, 1, self.n_devices)),
                     (0, 1, 2, 3))
    annotation = '{devices=[1,1,1,%d]%s}' % (self.n_devices, ','.join([
        str(i) for i in range(self.n_devices)
    ])) if self.n_devices > 1 else '{maximal device=0}'
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt))

    actual = (xt + xt).cpu()
    self.assertTrue(torch.allclose(expected, actual))

  def test_clear_sharding(self):
    xt = torch.randn(2, 4, 8, 16).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, 1, 1, self.n_devices)),
                     (0, 1, 2, 3))
    self.assertTrue(torch_xla._XLAC._get_xla_sharding_spec(xt))
    xs.clear_sharding(xt)
    self.assertFalse(torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_deep_copy(self):
    xt = torch.randn(2, 4, 8, 16).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, 1, 1, self.n_devices)),
                     (0, 1, 2, 3))
    xt2 = copy.deepcopy(xt)
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(xt),
        torch_xla._XLAC._get_xla_sharding_spec(xt2))

  def test_mark_step_with_sharding(self):
    xt = torch.ones(2, 2).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, self.n_devices)), (0, 1))
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(xt)
    xm.mark_step()  # mark_step should preserve the sharding
    self.assertEqual(sharding_spec, torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_inplace_add_with_sharding(self):
    xt = torch.ones(2, 2).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, self.n_devices)), (0, 1))
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(xt)
    xt.add_(1)  # inplace update should preserve the sharding
    self.assertEqual(sharding_spec, torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_shard_hashing(self):
    xt1 = torch.ones(2, 2).to(xm.xla_device())
    xt2 = torch.ones(2, 2).to(xm.xla_device())

    # Add sharding to xt1, this should result in the hashes being different for
    # xt1 and xt2
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), (0, 1))

    # Adding 0 to the tensor force graph compilation, which would catch IR hash
    # collisions
    self.assertTrue(torch.allclose(xt1 + 0, xt2 + 0))

    # Check that hashes are different for the sharded and non-sharded tensors
    hash1 = torch_xla._XLAC._get_graph_hash([xt1])
    hash2 = torch_xla._XLAC._get_graph_hash([xt2])
    self.assertNotEqual(hash1, hash2)

    # Adding 0 to the tensor force graph compilation, which would catch IR hashi
    # collisions
    self.assertTrue(torch.allclose(xt1 + 0, xt2 + 0))


class VirtualDeviceTest(XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    os.environ["XLA_USE_SPMD"] = "1"
    super().setUpClass()

  def test_mark_sharding(self):
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), partition_spec)
    self.assertTrue(
        torch.allclose(
            xt1 + 0,
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8],
                         dtype=torch.float,
                         device=xm.xla_device())))

  def test_metrics_recorded(self):
    met.clear_counters()
    partition_spec = (0, 1)
    xt1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                       dtype=torch.float,
                       device=xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), partition_spec)
    self.assertIn("VirtualDeviceUsage", met.counter_names())
    self.assertNotEqual(met.counter_value("VirtualDeviceUsage"), 0)

  def test_model_weight_metrics(self):
    met.clear_counters()
    partition_spec = (0, 1)
    model = nn.Linear(128, 64).to(xm.xla_device())
    xs.mark_sharding(model.weight, self._get_mesh((1, self.n_devices)),
                     partition_spec)
    self.assertIn("VirtualDeviceUsage", met.counter_names())
    self.assertNotEqual(met.counter_value("VirtualDeviceUsage"), 0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
