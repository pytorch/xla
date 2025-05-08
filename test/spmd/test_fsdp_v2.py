import copy
import functools
import unittest
import os
import sys

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

import test_xla_sharding_base
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2


# TODO(alanwaketan): Add more tests for FSDPv2.
class FSDPv2Test(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  def test_fsdp_v2_basic(self):
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    model.fc1 = FSDPv2(model.fc1, mesh=mesh)
    model.fc2 = FSDPv2(model.fc2, mesh=mesh)
    model = FSDPv2(model, mesh=mesh)

    # Make sure all weights are sharded.
    if self.n_devices > 1:
      annotation = '{devices=[1,%d]%s}' % (self.n_devices, ','.join(
          [str(i) for i in range(self.n_devices)]))
      self.assertEqual(annotation,
                       torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))
      self.assertEqual(annotation,
                       torch_xla._XLAC._get_xla_sharding_spec(model.fc2.weight))

    x = torch.randn(16, 128).to(xm.xla_device())
    xs.mark_sharding(x, mesh, ('fsdp', None))
    output = model(x)
    # Make sure output are sharded.
    if self.n_devices > 1:
      annotation = '{devices=[%d,1]%s}' % (self.n_devices, ','.join(
          [str(i) for i in range(self.n_devices)]))
      self.assertEqual(annotation,
                       torch_xla._XLAC._get_xla_sharding_spec(output))

    loss = output.sum()
    loss.backward()

    # Make sure optimization barrier is applied.
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([model.fc2.weight.grad])
    self.assertIn(
        'opt-barrier.38 = (f32[1,64]{0,1}, f32[1]{0}, f32[16,64]{1,0}) opt-barrier((f32[1,64]{0,1}, f32[1]{0}, f32[16,64]{1,0}) %tuple.37',
        hlo)

    # Make sure the model can execute without error.
    torch_xla.sync()
    xm.wait_device_ops()

  def test_fsdp_v2_output_correctness(self):
    model_expected = self.SimpleLinear().to(xm.xla_device())

    model = copy.deepcopy(model_expected)
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    model.fc1 = FSDPv2(model.fc1, mesh=mesh)
    model.fc2 = FSDPv2(model.fc2, mesh=mesh)
    model = FSDPv2(model, mesh=mesh)

    x_expected = torch.randn(16, 128).to(xm.xla_device())

    x = copy.deepcopy(x_expected)
    xs.mark_sharding(x, mesh, ('fsdp', None))

    output_expected = model_expected(x_expected)
    output = model(x)
    self.assertTrue(torch.allclose(output_expected.cpu(), output.cpu()))

  def test_fsdp_v2_auto_wrap_basic(self):
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={torch.nn.Linear},
    )
    model = FSDPv2(model, mesh=mesh, auto_wrap_policy=auto_wrap_policy)

    self.assertTrue(isinstance(model.fc1, FSDPv2))
    self.assertTrue(isinstance(model.fc2, FSDPv2))

  def test_fsdp_v2_auto_wrap_callable(self):
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={torch.nn.Linear},
    )

    def auto_wrapper_callable(m, *args, **kwargs):
      # Does nothing.
      return m

    model = FSDPv2(
        model,
        mesh=mesh,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=auto_wrapper_callable)

    # Since the callable is doing nothing, the children should not be wrapped.
    self.assertFalse(isinstance(model.fc1, FSDPv2))
    self.assertFalse(isinstance(model.fc2, FSDPv2))

  def test_fsdp_v2_global_mesh(self):
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    xs.set_global_mesh(mesh)

    model = FSDPv2(model)
    self.assertEqual(id(model._mesh), id(mesh))

  def test_fsdp_v2_global_mesh_error(self):
    model = self.SimpleLinear().to(xm.xla_device())
    xs.set_global_mesh(None)

    with self.assertRaises(ValueError):
      model = FSDPv2(model)

  def test_fsdp_v2_cpu_model(self):
    cpu_model = self.SimpleLinear()

    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    xs.set_global_mesh(mesh)

    model = FSDPv2(cpu_model)
    self.assertEqual(
        str(list(model._orig_module.parameters())[0].device), "xla:0")

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_fsdp_v2_multi_slice(self):
    model = self.SimpleLinear().to(xm.xla_device())
    mesh = self._get_mesh((2, self.n_devices // 2, 1), None,
                          ('data', 'fsdp', 'tensor'))
    model = FSDPv2(model, mesh=mesh, extra_data_axis="data")

    # Make sure all weights are sharded.
    annotation = '{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}'
    if self.n_devices == 8:
      annotation = '{devices=[1,4,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}'
    self.assertEqual(annotation,
                     torch_xla._XLAC._get_xla_sharding_spec(model.fc1.weight))
    self.assertEqual(annotation,
                     torch_xla._XLAC._get_xla_sharding_spec(model.fc2.weight))

    x = torch.randn(16, 128).to(xm.xla_device())
    xs.mark_sharding(x, mesh, (('data', 'fsdp'), None))
    output = model(x)
    # Make sure output are sharded.
    annotation = '{devices=[4,1]0,1,2,3}'
    if self.n_devices == 8:
      annotation = '{devices=[8,1]0,1,2,3,4,5,6,7}'
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(x))
    self.assertEqual(annotation, torch_xla._XLAC._get_xla_sharding_spec(output))

    # Make sure the model can execute without error.
    torch_xla.sync()
    xm.wait_device_ops()

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_fsdp_v2_multi_slice_output_correctness(self):
    model_expected = self.SimpleLinear().to(xm.xla_device())

    model = copy.deepcopy(model_expected)
    mesh = self._get_mesh((2, self.n_devices // 2, 1), None,
                          ('data', 'fsdp', 'tensor'))
    model = FSDPv2(model, mesh=mesh, extra_data_axis="data")

    x_expected = torch.randn(16, 128).to(xm.xla_device())

    x = copy.deepcopy(x_expected)
    xs.mark_sharding(x, mesh, (('data', 'fsdp'), None))

    output_expected = model_expected(x_expected)
    output = model(x)
    self.assertTrue(torch.allclose(output_expected.cpu(), output.cpu()))

  def test_fsdp_v2_multi_slice_error(self):
    model = self.SimpleLinear().to(xm.xla_device())
    xs.set_global_mesh(
        self._get_mesh((2, self.n_devices // 2, 1), None,
                       ('data', 'fsdp', 'tensor')))

    with self.assertRaisesRegex(ValueError,
                                "The provided ddp axis is not in the mesh."):
      model = FSDPv2(model, extra_data_axis='ddp')


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
