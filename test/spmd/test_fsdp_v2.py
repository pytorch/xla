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
    model.fc1 = FSDPv2(model.fc1, mesh)
    model.fc2 = FSDPv2(model.fc2, mesh)
    model = FSDPv2(model, mesh)

    # Make sure all weights are sharded.
    if self.n_devices > 1:
      annotation = '{devices=[%d,1]%s}' % (self.n_devices, ','.join(
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
    xm.mark_step()
    xm.wait_device_ops()

  def test_fsdp_v2_output_correctness(self):
    model_expected = self.SimpleLinear().to(xm.xla_device())

    model = copy.deepcopy(model_expected)
    mesh = self._get_mesh((self.n_devices, 1), None, ('fsdp', 'tensor'))
    model.fc1 = FSDPv2(model.fc1, mesh)
    model.fc2 = FSDPv2(model.fc2, mesh)
    model = FSDPv2(model, mesh)

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
    model = FSDPv2(model, mesh, auto_wrap_policy=auto_wrap_policy)

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
        mesh,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=auto_wrapper_callable)

    # Since the callable is doing nothing, the children should not be wrapped.
    self.assertFalse(isinstance(model.fc1, FSDPv2))
    self.assertFalse(isinstance(model.fc2, FSDPv2))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
