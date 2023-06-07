import copy

import unittest
import math
import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
import test_xla_sharding_base


class BasicShardingTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    os.environ["XLA_USE_SPMD"] = "1"
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

  def test_xla_shards(self):
    num_element = self.n_devices
    mesh = self._get_mesh((self.n_devices,))
    t = torch.arange(num_element, dtype=torch.float32)
    xt = xs.mark_sharding(t.to(xm.xla_device()), mesh, (0,))

    shards = xt.local_shards
    self.assertEqual(len(shards), self.n_devices)
    shard_len = math.ceil(num_element / self.n_devices)
    for i, shard in enumerate(shards):
      self.assertEqual(shard.data.device, torch.device('cpu'))
      self.assertEqual(shard.data.shape, (shard_len,))
      start, end = (i, i + 1) * shard_len
      expected = torch.arange(start, end, dtype=torch.float32)
      self.assertTrue(torch.allclose(shard.data, expected))
      if isinstance(shard.indices, list):
        self.assertEqual(len(shard.indices), len(t.shape))
        self.assertEqual(shard.indices[0], slice(start, end, 1))
      else:
        self.assertIsInstance(shard.indices, type(Ellipsis))
      self.assertTrue(torch.allclose(shard.data, t[shard.indices]))

  def test_padded_xla_shards(self):
    num_element = self.n_devices + 1  # Ensure padding with two or more devices
    mesh = self._get_mesh((self.n_devices,))
    t = torch.arange(num_element, dtype=torch.float32)
    xt = xs.mark_sharding(t.to(xm.xla_device()), mesh, (0,))
    shards = xt.local_shards
    self.assertEqual(len(shards), self.n_devices)
    shard_len = math.ceil(num_element / self.n_devices)
    for i, shard in enumerate(shards):
      self.assertEqual(shard.data.device, torch.device('cpu'))
      self.assertEqual(shard.data.shape, (shard_len,))
      # Tensor shards will be zero-padded
      start, end = min(i * shard_len, t.shape[0]), min((i + 1) * shard_len,
                                                       t.shape[0])
      if start < num_element:
        expected = torch.arange(start, end, dtype=torch.float32)
        pad_len = shard_len - expected.shape[0]
        expected = F.pad(expected, (0, pad_len), "constant", 0)
      else:
        expected = torch.zeros(shard.data.shape, dtype=torch.float32)
      self.assertTrue(torch.allclose(shard.data, expected))
      if isinstance(shard.indices, list):
        self.assertEqual(len(shard.indices), len(t.shape))
        self.assertEqual(shard.indices[0], slice(start, end, 1))
      else:
        self.assertIsInstance(shard.indices, type(Ellipsis))
      self.assertTrue(torch.allclose(shard.unpadded_data, t[shard.indices]))

  def test_replicated_xla_shards(self):
    num_element = self.n_devices
    mesh = self._get_mesh((self.n_devices,))
    t = torch.arange(num_element, dtype=torch.float32)
    xt = xs.mark_sharding(t.to(xm.xla_device()), mesh, (None,))
    shards = xt.local_shards
    self.assertEqual(len(shards), self.n_devices)
    for i, shard in enumerate(shards):
      self.assertEqual(shard.data.device, torch.device('cpu'))
      self.assertEqual(shard.data.shape, (num_element,))
      self.assertTrue(torch.allclose(shard.data, t))
      self.assertIsInstance(shard.indices, type(Ellipsis))
      self.assertTrue(torch.allclose(shard.data, t[shard.indices]))
      self.assertTrue(torch.allclose(shard.data, shard.unpadded_data))

  def test_load_local_shards(self):
    num_element = self.n_devices
    mesh = self._get_mesh((self.n_devices,))
    t = torch.arange(num_element, dtype=torch.float32) + 1
    xt = xs.mark_sharding(t.to(xm.xla_device()), mesh, (0,))
    local_shards = xt.local_shards
    self.assertTrue(len(local_shards) == self.n_devices)

    # More than one device is required for sharding to not be REPLICATED
    if self.n_devices > 1:
      for shard in local_shards:
        # Update the shard's data on CPU
        self.assertEqual(shard.data.device, torch.device('cpu'))
        shard.data *= -1
      # Loading a complete list of shards should succeed
      xt.load_local_shards_(local_shards)
      self.assertTrue(torch.allclose(xt.cpu(), -t))

    # Loading an incomplete list of shards should fail
    with self.assertRaises(RuntimeError):
      xt.load_local_shards_(local_shards[:-1])

    # Loading incompatible shapes should fail
    for local_shard in local_shards:
      local_shard.data = torch.randn(*(2 * local_shard.data.shape))
    with self.assertRaises(RuntimeError):
      xt.load_local_shards_(local_shards)

    # Replicated shards should fail
    rt = xs.mark_sharding(t.to(xm.xla_device()), mesh, (None,))
    local_shards = rt.local_shards
    with self.assertRaises(RuntimeError):
      rt.load_local_shards_(local_shards)

  def test_xla_sharding_type(self):
    t = torch.randn(10, 20).to(xm.xla_device())
    self.assertEqual(torch_xla._XLAC._get_xla_sharding_type(t), None)

    x_dim = 2 if self.n_devices % 4 == 0 else 1
    mesh = self._get_mesh((x_dim, self.n_devices // x_dim))
    xt = xs.mark_sharding(t, mesh, (0, 1))
    if self.n_devices > 1:
      self.assertEqual(xt.sharding_type, xs.ShardingType.TILED)
    else:
      self.assertEqual(xt.sharding_type, xs.ShardingType.REPLICATED)

    xs.clear_sharding(t)
    xt = xs.mark_sharding(t, mesh, (None, None))
    self.assertEqual(xt.sharding_type, xs.ShardingType.REPLICATED)

    xs.clear_sharding(t)
    xt = xs.mark_sharding(t, mesh, (None, 1))
    if self.n_devices > 1:
      self.assertEqual(xt.sharding_type, xs.ShardingType.PARTIAL)
    else:
      self.assertEqual(xt.sharding_type, xs.ShardingType.REPLICATED)

  def test_custom_tile_assignment(self):
    xt = torch.randn(10, 20).to(device=xm.xla_device())
    mesh_shape = (1, self.n_devices)
    device_ids = np.flip(self.device_ids)
    mesh = self._get_mesh(mesh_shape, device_ids)
    xs.mark_sharding(xt, mesh, (0, 1))

    if self.n_devices > 1:
      annotation = '{devices=[1,%d]%s}' % (self.n_devices, ','.join(
          [str(i) for i in reversed(range(self.n_devices))]))
      self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt))

  def test_mark_sharding_2d(self):
    t1 = torch.randn(1, 128, device='cpu')
    t2 = torch.randn(1, 128, device='cpu')
    expected = t1 + t2

    xt1 = t1.to(xm.xla_device())
    xt2 = t2.to(xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), (0, 1))

    if self.n_devices > 1:
      annotation = '{devices=[1,%d]%s}' % (self.n_devices, ','.join(
          [str(i) for i in range(self.n_devices)]))
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

    if self.n_devices > 1:
      annotation = '{devices=[1,1,%d,%d]%s}' % (
          z_dim, self.n_devices // z_dim, ','.join(
              [str(i) for i in range(self.n_devices)]))
      self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(xt))

    actual = (xt + xt).cpu()
    self.assertTrue(torch.allclose(expected, actual))

  def test_mark_sharding_partial(self):
    device = xm.xla_device()
    xt1 = torch.randn(4, 4).to(xm.xla_device())
    xt2 = torch.randn(4, 4).to(xm.xla_device())
    expected = (xt1 @ xt2).cpu()

    # Shard along two axes if four or more devices are available
    z_dim = 2 if self.n_devices >= 4 else 1
    mesh = self._get_mesh((z_dim, self.n_devices // z_dim))
    xs.mark_sharding(xt1, mesh, (0, None))

    # partial replication requires >1 devices; otherwise, it's replicated.
    if self.n_devices > 1:
      # xt1 is sharded `z_dim`-way, replicated `n_devices/z_dim`-way.
      self.assertTrue('last_tile_dim_replicate' in
                      torch_xla._XLAC._get_xla_sharding_spec(xt1))
      self.assertTrue('[%d,1,%d]' %
                      (z_dim, self.n_devices //
                       z_dim) in torch_xla._XLAC._get_xla_sharding_spec(xt1))
    actual = (xt1 @ xt2).cpu()
    self.assertTrue(torch.allclose(expected, actual))

  def test_partial_replication_addmm(self):
    device = xm.xla_device()
    z_dim = 2 if self.n_devices >= 4 else 1
    mesh = self._get_mesh((z_dim, self.n_devices // z_dim))

    xx = torch.randn(16, 128).to(device)
    xw = torch.randn(128, 256).to(device)
    xb = torch.randn(16, 256).to(device)
    expected = (xx @ xw + xb).cpu()

    xs.mark_sharding(xx, mesh, (0, None))
    xs.mark_sharding(xw, mesh, (None, 1))

    # Check if the partial replication annotations are passed to the compiler.
    # Note that partial replication requires >1 devices; otherwise, it's replicated.
    if self.n_devices > 1:
      self.assertTrue('last_tile_dim_replicate' in
                      torch_xla._XLAC._get_xla_sharding_spec(xx))
      self.assertTrue('last_tile_dim_replicate' in
                      torch_xla._XLAC._get_xla_sharding_spec(xw))
    actual = (xx @ xw + xb).cpu()
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

  def test_execute_replicated_metrics(self):
    met.clear_all()
    xt = torch.ones(2, 2).to(xm.xla_device())
    xs.mark_sharding(xt, self._get_mesh((1, self.n_devices)), (0, 1))
    xt += 2
    sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(xt)
    xm.mark_step()
    xm.wait_device_ops()
    self.assertEqual(met.metric_data('ExecuteReplicatedTime')[0], 1)

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
      if self.n_devices > 1:
        self.assertEqual(
            sharding_spec,
            torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))
      else:
        # single device execution defaults to implicit replication.
        self.assertFalse(
            torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))

  def test_sharding_propagation(self):
    met.clear_all()
    self.assertFalse(met.counter_value("ReplicateShardedData"))

    # Linear model with two linear layers and only one is annotated.
    model = self.SimpleLinear().to(xm.xla_device())
    xs.mark_sharding(model.fc1.weight, self._get_mesh((1, self.n_devices)),
                     (0, 1))
    self.assertTrue(torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))
    self.assertFalse(torch_xla._XLAC._get_xla_sharding_spec(model.fc2.weight))

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

    # Verify that the fc1 & output are sharded and valid
    model.fc1.weight.to('cpu')
    output.to('cpu')
    self.assertEqual(met.counter_value("ReplicateShardedData"), 2)

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

  def test_send_cpu_data_to_device_with_sharding(self):
    # Execute pending graph to avoid contaminating metrics
    xm.mark_step(wait=True)
    met.clear_all()

    tensor = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    mesh = self._get_mesh((1, self.n_devices))

    # Create a ShardingSpec and use it to shard the tensor while sending to
    # device
    sharding_spec = xs.ShardingSpec(mesh, (0, 1))
    self.assertTrue(sharding_spec.can_apply(tensor))
    xtensors = xm.send_cpu_data_to_device([tensor],
                                          xm.xla_device(),
                                          input_sharding=sharding_spec)
    self.assertEqual(len(xtensors), 1)
    outbound = met.metric_data("OutboundData")[1]
    self.assertEqual(outbound, tensor.element_size() * tensor.nelement())

    # Verify the resulting sharding annotation matches an explicit
    # `mark_sharding` call.
    xt = xtensors[0]
    explicit_xt = tensor.to(xm.xla_device())
    xs.mark_sharding(explicit_xt, mesh, (0, 1))
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(xt),
        torch_xla._XLAC._get_xla_sharding_spec(explicit_xt))

  def test_multiple_operations(self):
    t1 = torch.randn(2, 2)
    t2 = torch.randn(2, 2)
    expected_1 = t1 + t2
    xt1 = t1.to(xm.xla_device())
    xt2 = t2.to(xm.xla_device())
    xs.mark_sharding(xt1, self._get_mesh((1, self.n_devices)), (0, 1))
    xt3 = xt1 + xt2
    self.assertTrue(torch.allclose(expected_1, xt3.cpu()))

    t4 = torch.randn(2, 2)
    t5 = torch.randn(2, 2)
    expected_2 = t4 + t5
    xt4 = t4.to(xm.xla_device())
    xt5 = t5.to(xm.xla_device())
    xs.mark_sharding(xt4, self._get_mesh((1, self.n_devices)), (0, 1))
    xs.mark_sharding(xt5, self._get_mesh((1, self.n_devices)), (0, 1))
    xt6 = xt4 + xt5
    self.assertTrue(torch.allclose(expected_2, xt6.cpu()))

  def test_no_sharding(self):
    partition_spec = (0, 1)
    t1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                      dtype=torch.float,
                      device=xm.xla_device())
    t2 = torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]],
                      dtype=torch.float,
                      device=xm.xla_device())
    t3 = t1 + t2
    t3_expected = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    self.assertEqual(t3.tolist()[0], t3_expected)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
