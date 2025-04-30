import torch
import torch.nn as nn
import torch_xla
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
from torch_xla.distributed.fsdp.wrap import (
    always_wrap_policy as always_wrap,)

import torch.distributed as dist
import torch_xla.core.xla_model as xm

import sys

_TORCHDISTX_AVAIL = True
try:
  from torchdistx import deferred_init
except ImportError:
  _TORCHDISTX_AVAIL = False


def _reset_params_if_meta(is_meta, model):
  # For torchdistX init, we don't need to call reset_params, as
  # deferred_init(model).materialize() is equivalent to model().
  if is_meta:
    model.reset_parameters()


class MyLinear(nn.Linear):
  """
    Linear layer with deterministic reset_parameters for testing.
    """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def reset_parameters(self, *args, **kwargs):
    with torch.no_grad():
      self.weight.fill_(1)


class MyModel(nn.Module):

  def __init__(self, device):
    super().__init__()
    self.lin1 = MyLinear(2, 2, bias=False, device=device)
    self.lin2 = MyLinear(2, 2, bias=False, device=device)

  def forward(self, x):
    return self.lin2(self.lin1(x))

  def reset_parameters(self, *args, **kwargs):
    for m in [self.lin1, self.lin2]:
      if not isinstance(m, XlaFullyShardedDataParallel):
        m.reset_parameters()


def _init_with_reset_params(module):
  """
    to_empty + reset_parameters() init function example for modules
    initailized with device="meta"
    """
  is_meta = any(t.is_meta for t in module.parameters())
  if is_meta:
    module.to_empty(device=xm.xla_device())
  with torch.no_grad():
    module.reset_parameters()


def _init_with_torchdistX(module):
  """
    torchdistX-based deferred module initialization function example
    using ``materialize_module``.
    """
  assert _TORCHDISTX_AVAIL

  def check_fn(k):
    return not isinstance(k, XlaFullyShardedDataParallel)

  deferred_init.materialize_module(module, check_fn=check_fn)


class TestFSDPWithMetaDevice():

  def _compare_fsdp(self, fsdp1, fsdp2):
    for p1, p2 in zip(fsdp1.parameters(), fsdp2.parameters()):
      assert (torch.allclose(p1.cpu(), p2.cpu()))

  def _test_simple_model_with_meta_device(self, meta_module_fn, init_fn=None):
    # Create model on meta device and wrap with FSDP.
    model = meta_module_fn()
    inp = torch.randn(10, 2, device=xm.xla_device())

    fsdp_meta = XlaFullyShardedDataParallel(
        model,
        auto_wrap_policy=always_wrap,
        param_init_fn=init_fn,
    )
    meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=1e-3)
    fsdp_meta(inp).sum().backward()
    meta_opt.step()
    torch_xla.sync()

    regular = MyModel(device=xm.xla_device())
    fsdp_regular = XlaFullyShardedDataParallel(
        regular, auto_wrap_policy=always_wrap)
    regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=1e-3)
    fsdp_regular(inp).sum().backward()
    regular_opt.step()
    torch_xla.sync()

    self._compare_fsdp(fsdp_meta, fsdp_regular)

  def test_simple_model_with_meta_device_reset_params(self):

    def meta_module_fn():
      return MyModel(device="meta")

    self._test_simple_model_with_meta_device(meta_module_fn,
                                             _init_with_reset_params)

  def test_simple_model_with_meta_default_reset_params(self):

    def meta_module_fn():
      return MyModel(device="meta")

    self._test_simple_model_with_meta_device(meta_module_fn)

  def test_simple_model_with_torchdistX_init_fn(self):

    def meta_module_fn():
      return deferred_init.deferred_init(MyModel, device=xm.xla_device())

    self._test_simple_model_with_meta_device(
        meta_module_fn, init_fn=_init_with_torchdistX)

  def test_simple_model_with_default_torchdistX(self):

    def meta_module_fn():
      return deferred_init.deferred_init(MyModel, device=xm.xla_device())

    self._test_simple_model_with_meta_device(meta_module_fn)


def _mp_fn(index):
  device = xm.xla_device()
  # This test fails on GPU with 03/30 TF-pin update (https://github.com/pytorch/xla/pull/4840)
  if xm.xla_device_hw(device) in ('TPU', 'NEURON'):
    dist.init_process_group('xla', init_method='xla://')
    test = TestFSDPWithMetaDevice()
    test.test_simple_model_with_meta_device_reset_params()
    test.test_simple_model_with_meta_default_reset_params()
    if _TORCHDISTX_AVAIL:
      test.test_simple_model_with_torchdistX_init_fn()
      test.test_simple_model_with_default_torchdistX()
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
