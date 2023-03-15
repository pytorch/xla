import copy

import unittest
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
import test_xla_sharding_base


class BasicShardingTest(test_xla_sharding_base.XlaShardingTest):

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
    # Shard along two axes if four or more devices are available
    z_dim = 2 if self.n_devices >= 4 else 1
    xs.mark_sharding(xt, self._get_mesh((1, 1, z_dim, self.n_devices // z_dim)),
                     (0, 1, 2, 3))
    annotation = '{devices=[1,1,%d,%d]%s}' % (
        z_dim, self.n_devices // z_dim,
        ','.join([str(i) for i in range(self.n_devices)
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

  def test_clone(self):
    xt = torch.randn(2, 4, 8, 16).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, 1, 1, self.n_devices)),
                     (0, 1, 2, 3))
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(xt)
    xt2 = xt.clone()

    # check the original sharding spec is preserved after clone()
    self.assertEqual(sharding_spec, torch_xla._XLAC._get_xla_sharding_spec(xt))

    # check the cloned sharding spec is the same
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(xt),
        torch_xla._XLAC._get_xla_sharding_spec(xt2))

  def test_mark_step_with_sharding(self):
    xt = torch.ones(2, 2).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, self.n_devices)), (0, 1))
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(xt)
    xm.mark_step()  # mark_step should preserve the sharding
    self.assertEqual(sharding_spec, torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_optimizer_step_with_sharding(self):
    # Use simple linear model to test model parameter sharding
    model = self.SimpleLinear().to(xm.xla_device())
    xs.mark_sharding(model.fc1.weight, self._get_mesh((1, self.n_devices)),
                     (0, 1))
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    data = torch.randn(128, 128).to(xm.xla_device())
    target = torch.zeros(128).to(xm.xla_device())
    loss_fn = nn.CrossEntropyLoss()
    for i in range(3):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      xm.mark_step()
      # Sharding is persisted across mark_step calls, and test if the sharded computation
      # can repeat more than once without crashing.
      self.assertEqual(sharding_spec,
                       torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))

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

  def test_transfer_sharded_data_to_host(self):
    xt1 = torch.ones(16, 16).to(xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), (0, 1))
    t1 = xt1.cpu()
    self.assertTrue(torch.allclose(t1, torch.ones(16, 16)))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
