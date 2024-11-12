# Parse local options first, and rewrite the sys.argv[].
# We need to do that before import "common", as otherwise we get an error for
# unrecognized arguments.
import argparse
import os
import sys
import tempfile

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--replicated', action='store_true')
parser.add_argument('--long_test', action='store_true')
parser.add_argument('--max_diff_count', type=int, default=25)
parser.add_argument('--verbosity', type=int, default=0)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers
from absl.testing import absltest, parameterized

# Normal imports section starts here.
import collections
import copy
import itertools
import math
from numbers import Number
import numpy
import random
import re
import torch
import torch.autograd as ad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.testing._internal.common_device_type import dtypes
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    all_types_and,
)
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_op_registry as xor
import torch_xla.distributed.data_parallel as dp
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
import torch_xla.debug.metrics as met
import torch_xla.debug.model_comparator as mc
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr
import torch_xla.test.test_utils as xtu
import torch_xla.utils.dlpack as xdlpack
import torch_xla.utils.utils as xu
import torch_xla.utils.serialization as xser
import torch_xla.core.xla_model as xm
import torch_xla.core.functions as xf
import torch_xla.debug.profiler as xp
import unittest
import test_utils

DeviceSupport = collections.namedtuple('DeviceSupport', ['num_devices'])

XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


def _is_on_tpu():
  return 'XRT_TPU_CONFIG' in os.environ or xr.device_type() == 'TPU'


skipOnTpu = unittest.skipIf(_is_on_tpu(), 'Not supported on TPU')
skipOnEagerDebug = unittest.skipIf(torch_xla.experimental.is_eager_mode(),
                                   'skip on eager debug mode')


def _skipIfFunctionalization(value=True, reason=""):
  verb = "is" if value else "is not"
  reason = f" Reason: {reason}" if reason else ""
  return unittest.skipIf(
      XLA_DISABLE_FUNCTIONALIZATION is value,
      f'Works only when functionalization {verb} disabled.{reason}.')


def skipIfFunctionalizationEnabled(reason):
  return _skipIfFunctionalization(value=False, reason=reason)


def skipIfFunctionalizationDisabled(reason):
  return _skipIfFunctionalization(value=True, reason=reason)


def onlyOnCUDA(fn):
  accelerator = os.environ.get("PJRT_DEVICE").lower()
  return unittest.skipIf(accelerator != "cuda", "PJRT_DEVICE=CUDA required")(fn)


def onlyIfXLAExperimentalContains(feat):
  experimental = os.environ.get("XLA_EXPERIMENTAL", "").split(":")
  return unittest.skipIf(feat not in experimental,
                         f"XLA_EXPERIMENTAL={feat} required")


def _gen_tensor(*args, **kwargs):
  return torch.randn(*args, **kwargs)


def _gen_int_tensor(*args, **kwargs):
  return torch.randint(*args, **kwargs)


def _gen_mask(size):
  return torch.randint(0, 2, size, dtype=torch.bool)


def _get_device_support(devname):
  devices = torch_xla._XLAC._xla_get_devices()
  num_devices = 0
  for device in devices:
    if re.match(devname + r':\d+$', device):
      num_devices += 1
  return DeviceSupport(num_devices=num_devices) if num_devices > 0 else None


def _support_replicated(devname, num_devices):
  devsup = _get_device_support(devname)
  if not devsup:
    return False
  return devsup.num_devices >= num_devices


def _random_inputs(shapes, num_replicas=1):
  random_tensors = []
  for _ in range(0, num_replicas):
    replica_inputs = []
    for shape in shapes:
      replica_inputs.append(_gen_tensor(*shape))
    random_tensors.append(tuple(replica_inputs))
  return tuple(random_tensors)


def _random_like(tensor_list):
  random_tensors = []
  for o in tensor_list:
    if o.dtype == torch.float32 or o.dtype == torch.float64:
      random_tensors += [_gen_tensor(*o.shape, dtype=o.dtype)]
    elif o.dtype == torch.int64:
      # TODO remove this, we shouldn't be needing to pass random_tensor for long types
      random_tensors += [torch.empty_like(o)]
    else:
      raise RuntimeError('Unsupported type: ', o.dtype)
  return random_tensors


def _zeros_like(tensor_list):
  zeros_tensors = []
  for o in tensor_list:
    if o.dtype == torch.float32 or o.dtype == torch.float64:
      zeros_tensors += [torch.zeros(*o.shape, dtype=o.dtype)]
    elif o.dtype == torch.int64:
      # TODO remove this, we shouldn't be needing to pass zeros_tensor for long types
      zeros_tensors += [torch.zeros_like(o)]
    else:
      raise RuntimeError('Unsupported type: ', o.dtype)
  return zeros_tensors


def onlyIfTorchSupportsCUDA(fn):
  return unittest.skipIf(
      not torch.cuda.is_available(), reason="requires PyTorch CUDA support")(
          fn)


def onlyIfPJRTDeviceIsCUDA(fn):
  return unittest.skipIf(
      os.environ.get("PJRT_DEVICE") not in ("GPU", "CUDA"),
      reason="requires CUDA as PJRT_DEVICE")(
          fn)


class TestToXlaTensorArena(test_utils.XlaTestCase):

  def test(self):
    xla_device = xm.xla_device()

    kdata = [_gen_tensor(2, 3), _gen_tensor(3, 4)]
    kdata.append([_gen_tensor(2, 5), _gen_tensor(3, 6)])
    data = dict()
    data[_gen_tensor(2, 2)] = tuple(kdata)
    data[_gen_tensor(2, 4)] = set([12.0, _gen_tensor(3, 7)])
    data['ABC'] = _gen_tensor(4, 3)

    def select_fn(v):
      return type(v) == torch.Tensor

    def convert_fn(tensors):
      devices = [str(xla_device)] * len(tensors)
      return torch_xla._XLAC._xla_tensors_from_aten(tensors, devices)

    def check_fn(v):
      if select_fn(v):
        return xm.is_xla_tensor(v)
      elif isinstance(v, (list, tuple, set)):
        for x in v:
          if not check_fn(x):
            return False
      elif isinstance(v, dict):
        for k, x in v.items():
          if not check_fn(k) or not check_fn(x):
            return False
      return True

    xla_data = xm.ToXlaTensorArena(convert_fn, select_fn).transform(data)
    self.assertTrue(check_fn(xla_data))


class TestParallelLoader(test_utils.XlaTestCase):

  def test(self):
    devices = [torch.device(x) for x in xm.get_xla_supported_devices()]
    A = 3.11
    B = 4.09
    batch_size = 128 * len(devices)
    gen = xu.FnDataGenerator(
        lambda x: x * A + B, batch_size, _gen_tensor, dims=[8], count=10)
    para_loader = pl.ParallelLoader(gen, devices)
    for device in devices:
      loader = para_loader.per_device_loader(device)
      for data, target in loader:
        self.assertEqual(data.device, device)
        self.assertEqual(target.device, device)


class TestAtenTensorTo(test_utils.XlaTestCase):

  def test(self):
    devices = xm.get_xla_supported_devices()
    for device in reversed(devices):
      t = _gen_tensor(8, 12)
      tto = t.to(device=torch.device(device))
      self.assertEqual(tto.device, torch.device(device))
    t = _gen_tensor(8, 12).to(device=torch.device(devices[0]))
    for device in devices[1:]:
      tto = t.to(device=torch.device(device))
      self.assertEqual(tto.device, torch.device(device))
    for i in range(0, len(devices) - 1):
      dev0 = devices[i]
      dev1 = devices[i + 1]
      t0 = torch.zeros(4, 4, device=torch.device(dev0))
      t1 = t0.to(device=torch.device(dev1))
      t0 = t0 + torch.ones_like(t0, device=torch.device(dev0))
      t1 = t1 + torch.ones_like(t1, device=torch.device(dev1))
      self.assertEqual(t0.cpu(), t1.cpu())


class XlaMNIST(nn.Module):

  def __init__(self):
    super(XlaMNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


@unittest.skipIf(
    xr.device_type() == 'CUDA',
    'Parallelism for DataParallel uses multi-threads. But cuda assumes one GPU device per process instead of relying on threads.'
)
class TestParallelTensorMNIST(test_utils.XlaTestCase):

  def test(self):
    # devices=['xla:0', 'xla:1', 'xla:2', 'xla:3'] for example.
    devices = xm.get_xla_supported_devices()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=8)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(batch_size, 1, 28,
                          28), torch.zeros(batch_size, dtype=torch.int64)),
        sample_count=sample_count * len(devices))

    def loop_fn(model, loader, device, context):
      loss_fn = nn.NLLLoss()
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

      for data, target in loader:
        with xu.TimedScope(msg='Training loop: ', printfn=None):
          optimizer.zero_grad()
          output = xu.timed(lambda: model(data), msg='Model: ', printfn=None)
          loss = xu.timed(
              lambda: loss_fn(output, target), msg='Loss: ', printfn=None)
          xu.timed(loss.backward, msg='LossBkw: ', printfn=None)
          xu.timed(
              lambda: xm.optimizer_step(optimizer), msg='Step: ', printfn=None)
          self.assertLess(loss.cpu().item(), 3.0)

    model_parallel = dp.DataParallel(XlaMNIST, device_ids=devices)
    model_parallel(loop_fn, train_loader)


class TestLongGraphChain(test_utils.XlaTestCase):

  def test(self):
    device = xm.xla_device()
    orig_x = torch.Tensor([[1, 2], [3, 4]])
    orig_y = torch.Tensor([[0.1, 0.2], [0.3, 0.4]])
    x = orig_x
    y = orig_y
    xla_x = orig_x.to(device)
    xla_y = orig_y.to(device)
    for i in range(0, 2000):
      x = x + 2 * y
      xla_x = xla_x + 2 * xla_y
    self.assertEqualRel(
        x,
        xla_x.cpu(),
        rel_err=1e-3,
        abs_err=5,
        max_diff_count=FLAGS.max_diff_count)


class TestSelect(test_utils.XlaTestCase):

  def test_get_xla_tensor(self):
    x = _gen_tensor(14, 24, 8, device=xm.xla_device())
    t = x.data.cpu()
    sx = x.select(1, 12)
    tx = t.select(1, 12)
    self.assertEqual(tx, sx.data.cpu())

  def test_masked_fill_scalar(self):

    def fn(tensor):
      # Build a mask from the first line of tensor.
      # Also, make it have the same rank as the original tensor.
      mask = tensor[0].ge(0.5).unsqueeze(dim=0)
      # Call masked_fill.
      return tensor.masked_fill(mask, 10)

    x = _gen_tensor(2, 2, device=xm.xla_device())
    x_cpu = x.cpu()
    self.assertEqual(fn(x_cpu), fn(x))


class TestRandom(test_utils.XlaTestCase):

  def test_random_from_to_bool(self):
    for from_val, to_val in [[0, 1], [0, 2], [1, 2]]:
      x = _gen_tensor(10, device=xm.xla_device())
      x.random_(from_val, to_val)
      delta = 1
      self.assertTrue(from_val <= x.to(torch.int).min() < (from_val + delta))
      self.assertTrue((to_val - delta) <= x.to(torch.int).max() < to_val)


class TestBinaryCrossEntropyLimitValue(test_utils.XlaTestCase):

  def test_cross_entropy_loss(self):

    def test_fn(pred, target):
      lossfn = nn.BCELoss()
      return lossfn(pred, target)

    pred = torch.tensor(1.0)
    target = torch.tensor(1.0)
    for offset in [1, 0, 1e-8, 1e-7]:
      self.runAtenTest([pred - offset, target], test_fn)


class TestNllLossLimitValue(test_utils.XlaTestCase):

  def test_nll_loss_inf(self):

    def test_fn(logits, target):
      return nn.functional.nll_loss(logits, target)

    inf = float('inf')
    logits = torch.tensor([[1., inf], [-inf, 2.]])
    for target in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      target_tensor = torch.tensor(target)
      self.runAtenTest([logits, target_tensor], test_fn)

  def test_nll_loss_nan(self):

    def test_fn(logits, target):
      return nn.functional.nll_loss(logits, target)

    nan = float('nan')
    logits = torch.tensor([[1, nan], [nan, 2]])
    # Need to include both nan being labeled and nan not being labeled case
    for target in [[0, 0], [0, 1], [1, 0], [1, 1]]:
      target_tensor = torch.tensor(target)
      self.runAtenTest([logits, target_tensor], test_fn)


class TestInterOpSyncTensors(test_utils.XlaTestCase):

  def test_inter_op_sync(self):

    def test_fn(x):
      # logaddexp can be replaced with any op that does not have
      # xla lowering.
      y = torch.logaddexp(x, x)
      return torch.masked_select(y, y.eq(0))

    x = torch.tensor([1., 2., 3.])
    self.runAtenTest([x], test_fn)


class TestDynamicShape(test_utils.XlaTestCase):

  def test_nonzero_shape(self):
    x = torch.tensor((0, 1, 2, 0, 3, 4), device=xm.xla_device())
    x_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(
        torch.nonzero(x, as_tuple=False), 0)
    self.assertEqual(x_dim0_shape.item(), 4)

  def test_masked_select_shape(self):
    x = torch.tensor((0, 1, 2, 0, 3, 4), device=xm.xla_device())
    mask = x.ge(2)
    x_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(
        torch.masked_select(x, mask), 0)
    self.assertEqual(x_dim0_shape.item(), 3)

  def test_nonzero_cast(self):
    t1 = torch.ones(5, 2, device=xm.xla_device())
    # Result of the nonzero should be the index type. Currently
    # index type is s64 on cpu and gpu, but s32 on TPU. We should be
    # able to cast it to any other type without error.
    t2 = torch.nonzero(t1.int()).float()
    xm.mark_step()


class TestOptimizationBarrier(test_utils.XlaTestCase):

  def test_optimization_barrier_correctness(self):
    device = xm.xla_device()
    # only test optimization_barrier on TPU
    if xm.xla_device_hw(device) != 'TPU':
      return
    x = torch.randn(5, 5, device=device)
    y = torch.randn(5, 5, device=device)
    z = x + y
    xm.optimization_barrier_([x, y])
    self.assertEqual(z, x + y)


class TestDataType(test_utils.XlaTestCase):

  def test_mixed_dtype_tuple(self):

    def op_fn(a):
      return xb.Op.tuple((a, a.cast(xb.Type.BF16)))

    op = xor.register('test_mixed_dtype_tuple', op_fn)
    xla_device = xm.xla_device()
    a_tensor = torch.randn([2, 3]).to(xla_device)
    a_result, a_cast = op(a_tensor)
    self.assertEqual(a_result.dtype, torch.float)
    self.assertEqual(a_cast.dtype, torch.bfloat16)


class TestAtenXlaTensor(test_utils.XlaTestCase):

  def test_get_real_xla_devices(self):
    devices = xm.get_xla_supported_devices()
    xla_devices = torch_xla._XLAC._xla_real_devices(devices)
    for device, xdevice in zip(devices, xla_devices):
      self.assertIsNotNone(re.fullmatch(r'[A-Z]+:\d+$', xdevice))

  def test_negative_slice(self):
    t = _gen_tensor(32, 24, 32)
    x = t.to(xm.xla_device())
    t_slice = t[:, :, -1]
    x_slice = x[:, :, -1]
    self.assertEqual(t_slice.data, x_slice.data.cpu())

  def test_negative_cat(self):
    t = _gen_tensor(2, 5, 3)
    x = t.to(xm.xla_device())
    t_cat = torch.cat([t, t], -1)
    x_cat = torch.cat([x, x], -1)
    self.assertEqual(t_cat.data, x_cat.data.cpu())

  def test_cat_empty_tensor(self):
    t = _gen_tensor(2, 5, 3)
    empty_tensor = torch.Tensor()
    x = t.to(xm.xla_device())
    empty_tensor_xla = empty_tensor.to(xm.xla_device())
    t_cat = torch.cat([t, empty_tensor], 0)
    x_cat = torch.cat([x, empty_tensor_xla], 0)
    self.assertEqual(t_cat.data, x_cat.data.cpu())

  def test_nan_to_num_in_place(self):
    t = torch.tensor([float('nan'), float('nan'), -float('nan'), 3.14])

    def fn(x):
      x.nan_to_num_(1.0, 2.0, 3.0)
      return x

    self.runAtenTest(t, fn)

  @skipOnTpu
  def test_nan_to_num_in_place_with_inf(self):
    # Since TPU converts double to float (unlike CPU), the Inf entries are
    # expected to be different. Skipping tests for Inf entries.
    t = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])

    def fn(x):
      x.nan_to_num_(1.0, 2.0, 3.0)
      return x

    self.runAtenTest(t, fn)

  @skipOnTpu
  def test_amp_foreach_non_finite_check_and_unscale_(self):
    # Since TPU converts double to float (unlike CPU), the Inf entries are
    # expected to be different. Skipping tests for Inf entries.
    grads0 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    grads1 = torch.tensor([1.0, 2.0, float('nan'), 4.0], dtype=torch.float32)
    inv_scale = torch.tensor(0.2, dtype=torch.float32)
    found_inf = torch.tensor(0, dtype=torch.float32)
    grads_output0 = grads0 * inv_scale
    found_inf_output0 = torch.tensor(0, dtype=torch.float32)
    found_inf_output1 = torch.tensor(1, dtype=torch.float32)

    xla_device = xm.xla_device()
    xla_grads0 = grads0.to(xla_device)
    xla_inv_scale = inv_scale.to(xla_device)
    xla_found_inf = found_inf.to(xla_device)
    torch._amp_foreach_non_finite_check_and_unscale_([xla_grads0],
                                                     xla_found_inf,
                                                     xla_inv_scale)
    self.assertEqual(grads_output0, xla_grads0, prec=1e-4)
    self.assertEqual(found_inf_output0, xla_found_inf)

    xla_grads1 = grads1.to(xla_device)
    torch._amp_foreach_non_finite_check_and_unscale_([xla_grads1],
                                                     xla_found_inf,
                                                     xla_inv_scale)
    self.assertEqual(found_inf_output1, xla_found_inf)

  def test_masked_fill_with_tensor(self):
    input = _gen_tensor(2, 5, 4, 3)
    mask = _gen_mask(input.size())
    value = torch.tensor(42)
    xla_input = input.to(xm.xla_device())
    xla_mask = mask.to(xm.xla_device())
    xla_value = value.to(xm.xla_device())
    result = torch.masked_fill(input, mask, value)
    xla_result = torch.masked_fill(xla_input, xla_mask, xla_value)
    self.assertEqual(input.data, xla_input.data.cpu())
    self.assertEqual(result.data, xla_result.data.cpu())

  def test_masked_fill_in_out_place(self):

    def test_fn(a, b, m):
      ar = torch.masked_fill(a, m, 17.0)
      bi = b.masked_fill_(m, 21.0)
      return ar, bi

    t1 = _gen_tensor(5, 3)
    t2 = _gen_tensor(*t1.size())
    self.runAtenTest([t1, t2, _gen_mask(t1.size())], test_fn)

  def test_add_mixed_device(self):
    input = _gen_tensor(3, 800, 1066)
    xla_input = input.to(xm.xla_device())
    output = input + 2
    xla_output = xla_input + 2
    self.assertEqual(output.data, xla_output.data.cpu())

  def test_mul_mixed_device(self):
    input = _gen_tensor(3, 800, 1066)
    xla_input = input.to(xm.xla_device())
    output = input * 2
    xla_output = xla_input * 2
    self.assertEqual(output.data, xla_output.data.cpu())

  def test_sub_mixed_device(self):
    input = _gen_tensor(3, 800, 1066)
    xla_input = input.to(xm.xla_device())
    output = input - 2
    xla_output = xla_input - 2
    self.assertEqual(output.data, xla_output.data.cpu())

  def test_div_mixed_device(self):
    input = _gen_tensor(3, 800, 1066)
    xla_input = input.to(xm.xla_device())
    output = input / 2
    xla_output = xla_input / 2
    self.assertEqual(output.data, xla_output.data.cpu())

  def test_rand(self):
    x = torch.rand(3, 5, device=xm.xla_device())
    self.assertEqual(x.device.type, 'xla')

  def test_randperm(self):
    x = torch.randperm(3, device=xm.xla_device(), dtype=torch.int32)
    self.assertEqual(x.device.type, 'xla')

  def test_randn_like(self):
    shape = (5, 1, 1)
    x = torch.randn_like(torch.zeros(shape, device=xm.xla_device()))
    self.assertEqual(x.device.type, 'xla')

  def test_rand_like(self):
    shape = (5, 1, 1)
    x = torch.rand_like(torch.zeros(shape, device=xm.xla_device()))
    self.assertEqual(x.device.type, 'xla')

  def test_randint_like(self):
    shape = (5, 1, 1)
    x = torch.randint_like(
        torch.zeros(shape, device=xm.xla_device(), dtype=torch.uint8), 6, 10)
    self.assertEqual(x.device.type, 'xla')

  def test_no_storage(self):
    x = torch.randn(5, device=xm.xla_device())
    self.assertRaises(Exception, x.device)

  def test_slice_copy(self):
    a = torch.rand(3, 3, 3)
    xla_device = xm.xla_device()
    xla_a = a.to(xla_device)
    shape = (4, 4, 4)
    b = a.new(*shape).zero_()
    xla_b = xla_a.new(*shape).zero_()
    b[:a.shape[0], :a.shape[1], :a.shape[2]].copy_(a)
    xla_b[:a.shape[0], :a.shape[1], :a.shape[2]].copy_(xla_a)
    self.assertEqual(b.data, xla_b.data.cpu())

  def test_slice_assign(self):
    a = torch.rand(3, 3, 3)
    xla_device = xm.xla_device()
    xla_a = a.to(xla_device)
    shape = (4, 4, 4)
    b = a.new(*shape).zero_()
    xla_b = xla_a.new(*shape).zero_()
    b[0, :, :] = 1
    xla_b[0, :, :] = 1
    self.assertEqual(b.data, xla_b.data.cpu())

  def test_slice_stepped_assign(self):
    a = torch.ones((10, 4))
    xla_device = xm.xla_device()
    xla_a = a.to(xla_device)
    a[:, 0::2] = 2
    xla_a[:, 0::2] = 2
    self.assertEqual(a.data, xla_a.data.cpu())

  def test_slice_stepped_other_assign(self):
    a = torch.ones((10, 4))
    xla_device = xm.xla_device()
    xla_a = a.to(xla_device)
    a[:, 1::4] = 2
    xla_a[:, 1::4] = 2
    self.assertEqual(a.data, xla_a.data.cpu())

  def test_ailing_slice(self):
    xla_device = xm.xla_device()
    a = torch.ones((1000, 324)).to(xla_device)
    xla_a = a.to(xla_device)
    w = a[:, 2::4]
    xla_w = a[:, 2::4]
    dw = torch.clamp(w, max=3.1)
    xla_dw = torch.clamp(xla_w, max=3.1)
    self.assertEqual(w.data, xla_w.data.cpu())

  def test_slice_rnd_stepped_assign(self):
    xla_device = xm.xla_device()
    size = 10
    for s in range(0, size - 1):
      for e in range(1, size - s):
        a = torch.ones((3, size))
        xla_a = a.to(xla_device)
        a[:, s::e] = 2
        xla_a[:, s::e] = 2
        self.assertEqual(a.data, xla_a.data.cpu())

  def test_arange_nan(self):
    with self.assertRaisesRegex(RuntimeError, r'unsupported range'):
      a = torch.arange(-5, float('nan'), device=xm.xla_device())
    with self.assertRaisesRegex(RuntimeError, r'unsupported range'):
      a = torch.arange(float('nan'), 5, device=xm.xla_device())

  def test_empty_advanced_indexing(self):
    xla_device = xm.xla_device()
    base = torch.randn(2, 3, 4, 5)
    xla_base = base.to(device=xla_device)
    result = base[:, torch.empty(0, 6, dtype=torch.int64)]
    xla_result = xla_base[:, torch.empty(0, 6, dtype=torch.int64)]
    self.assertEqual(result, xla_result)

  @unittest.skip(
      "grad_input produces wrong results after functionalization. pytorch/pytorch#91199"
  )
  def test_empty_strided(self):
    xla_device = xm.xla_device()
    m = nn.Conv1d(4, 6, kernel_size=3, groups=2)
    a = torch.rand(2, 4, 6, requires_grad=True)
    xla_m = copy.deepcopy(m).to(xla_device)
    xla_a = a.clone().to(xla_device).detach()
    xla_a.requires_grad = True
    output = m(a)
    grad_input = torch.autograd.grad(
        output, (a,) + tuple(m.parameters()), output, create_graph=True)
    grad_grad_input = torch.autograd.grad(
        output.sum() + sum(map(lambda x: x.sum(), grad_input)),
        (a, output) + tuple(m.parameters()),
        retain_graph=True)
    xla_output = xla_m(xla_a)
    xla_grad_input = torch.autograd.grad(
        xla_output, (xla_a,) + tuple(xla_m.parameters()),
        xla_output,
        create_graph=True)
    xla_grad_grad_input = torch.autograd.grad(
        xla_output.sum() + sum(map(lambda x: x.sum(), xla_grad_input)),
        (xla_a, xla_output) + tuple(xla_m.parameters()),
        retain_graph=True)
    self.assertEqual(output, xla_output, prec=1e-4)
    self.assertEqual(grad_input, xla_grad_input, prec=1e-4)
    self.assertEqual(grad_grad_input, xla_grad_grad_input, prec=1e-4)

  def test_clamp(self):
    a = torch.randn(3, 3)
    xla_a = a.to(xm.xla_device())
    b = torch.clamp(a, max=3.4)
    xla_b = torch.clamp(xla_a, max=3.4)
    self.assertEqual(b.data, xla_b.data.cpu())

  def test_rrelu_module(self):
    xla_device = xm.xla_device()
    a = torch.rand(1, 2, 2, requires_grad=True)
    xla_a = a.to(xla_device).detach()
    xla_a.requires_grad = True

    m = nn.RReLU()
    xla_m = m.to(xla_device)

    output = m(a)
    xla_output = xla_m(xla_a)
    self.assertEqual(output, xla_output.cpu())

    output.sum().backward()
    xla_output.sum().backward()
    self.assertEqual(a.grad, xla_a.grad.cpu())

  def test_max_broadcast(self):
    xla_device = xm.xla_device()
    a = torch.rand(3, 1, 2)
    b = torch.rand(4, 2)
    c = torch.max(a, b)
    xla_a = a.to(xla_device)
    xla_b = b.to(xla_device)
    xla_c = torch.max(xla_a, xla_b)
    self.assertEqual(c.data, xla_c.data.cpu())

  def test_sgn(self):
    xla_device = xm.xla_device()
    t = torch.randn(2, 3, dtype=torch.cfloat)
    # Generate inf+infj
    t[0][0].real.div_(0)
    t[0][0].imag.div_(0)
    # Generate nan+nanj
    t[0][1] = 0
    t[0][1].real.div_(0)
    t[0][1].imag.div_(0)
    # Generate 0+0j
    t[1][0] = 0
    # Generate inf+0j
    t[1][1].real.div_(0)
    t[1][1] = t[1][1].real.abs()
    # Generate -inf+0j
    t[1][2].real.div_(0)
    t[1][2] = t[1][1].real.abs() * -1
    a = t.sgn()
    xla_a = t.to(xla_device).sgn()
    self.assertEqual(a.data, xla_a.data.cpu())

    t = torch.randn(2, 3, dtype=torch.float32)
    t[0][0].div_(0)
    t[0][1] = 0
    t[0][1].div_(0)
    t[1][0] = 0
    t[1][2].div_(0)
    t[1][2] = t[1][1].abs() * -1
    a = t.sgn()
    xla_a = t.to(xla_device).sgn()
    self.assertEqual(a.data, xla_a.data.cpu())

  @skipIfFunctionalizationDisabled("view_as_real unsupported")
  def test_view_as_real_c64(self):
    xla_device = torch_xla.device()
    x = torch.randn(4, dtype=torch.cfloat, device=xla_device)
    real = torch.view_as_real(x)
    self.assertEqual(real.dtype, torch.float32)
    # XLA type of the real needs to be f32 as well
    self.assertIn("f32[4,2]", torch_xla._XLAC._get_xla_tensor_debug_info(real))
    # HLO generated needs to have type f32 as well
    self.assertIn("f32[4,2]",
                  torch_xla._XLAC._get_xla_tensors_text([real]).split('\n')[-3])

  @skipIfFunctionalizationDisabled("view_as_real unsupported")
  def test_view_as_real_c128(self):
    xla_device = torch_xla.device()
    x = torch.randn(4, dtype=torch.cdouble, device=xla_device)
    real = torch.view_as_real(x)
    self.assertEqual(real.dtype, torch.float64)
    # XLA type of the real needs to be f32 as well
    self.assertIn("f64[4,2]", torch_xla._XLAC._get_xla_tensor_debug_info(real))
    # HLO generated needs to have type f32 as well
    self.assertIn("f64[4,2]",
                  torch_xla._XLAC._get_xla_tensors_text([real]).split('\n')[-3])

  def test_index_put(self):
    xla_device = xm.xla_device()
    a = torch.tensor([1, 1, 1, 1]).to(xla_device).to(dtype=torch.float32)
    b = torch.rand(4) > 0.1
    a[b] = 10
    vset = b.sum().item()
    self.assertEqual(a.sum().item(), 10.0 * vset + (4.0 - vset))

  @skipOnTpu
  def test_pow_integer_types(self):
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: torch.pow(x, 2))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: torch.pow(2, x))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: torch.pow(x, x))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: x.pow_(2))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: x.pow_(x))

  @skipOnTpu
  def test_matmul_integer_types(self):
    # all variance of matmul: dot/mv/mm/bmm
    self.runAtenTest((torch.randint(10, (2,)), torch.randint(10, (2,))),
                     lambda x, y: torch.matmul(x, y))
    self.runAtenTest((torch.randint(10, (3, 4)), torch.randint(10, (4,))),
                     lambda x, y: torch.matmul(x, y))
    self.runAtenTest((torch.randint(10, (10, 3, 4)), torch.randint(10, (4,))),
                     lambda x, y: torch.matmul(x, y))
    self.runAtenTest((torch.randint(10,
                                    (10, 3, 4)), torch.randint(10, (10, 4, 5))),
                     lambda x, y: torch.matmul(x, y))
    self.runAtenTest((torch.randint(10, (10, 3, 4)), torch.randint(10, (4, 5))),
                     lambda x, y: torch.matmul(x, y))

  @skipOnTpu
  def test_addmm_integer_types(self):
    self.runAtenTest((torch.randint(10, (2, 3)), torch.randint(
        10, (2, 3)), torch.randint(10, (3, 3))),
                     lambda x, y, z: torch.addmm(x, y, z))

  @skipOnTpu
  def test_baddmm_integer_types(self):
    self.runAtenTest(
        (torch.randint(10, (10, 3, 5)), torch.randint(10, (10, 3, 4)),
         torch.randint(10, (10, 4, 5))), lambda x, y, z: torch.baddbmm(x, y, z))

  def test_view_empty(self):
    # These used to throw floating point exception.
    empty = torch.empty(0, device=xm.xla_device())
    with self.assertRaisesRegex(
        RuntimeError, r'unspecified dimension size -1 can be any value'):
      empty.view(-1, 0)
    with self.assertRaisesRegex(
        RuntimeError, r'unspecified dimension size -1 can be any value'):
      empty.view(3, 0, -1, 0)

  def test_view_1718(self):

    def test_fn(device):
      torch.manual_seed(0)
      linear = nn.Linear(8, 16).to(device=device)
      batch = torch.rand(4, 8).to(device=device)
      x = linear(batch)
      x[:, :4] = 0
      loss = x.sum()
      loss.backward()
      return loss, linear.weight.grad

    cpu_loss, cpu_weight_grad = test_fn('cpu')
    xla_loss, xla_weight_grad = test_fn(xm.xla_device())
    self.assertEqual(cpu_loss, xla_loss)
    self.assertEqual(cpu_weight_grad, xla_weight_grad)

  def test_inplace_view_backprop_base(self):
    root = torch.randn(2, 2, device=xm.xla_device(), requires_grad=True)
    x = root.clone()
    v1 = x.narrow(0, 0, 1)
    v1.mul_(2)
    x.sum().backward()
    self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

  def test_inplace_view_backprop_view_of_view(self):
    root = torch.randn(2, 2, device=xm.xla_device(), requires_grad=True)
    x = root.clone()
    v1 = x.narrow(0, 0, 1)
    v2 = x.narrow(0, 0, 1)
    v1.mul_(2)
    v2.sum().backward()
    self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

  def test_inplace_view_of_view(self):
    # modify view-of-view and backprop through base
    root = torch.randn(2, 2, device=xm.xla_device(), requires_grad=True)
    x = root.clone()
    v1 = x.narrow(0, 0, 1)
    v2 = v1.narrow(1, 1, 1)
    v2.mul_(2)
    x.sum().backward()
    self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

  def test_inplace_view_multiple_outputs(self):
    root = torch.arange(
        9., device=xm.xla_device()).reshape(3, 3).requires_grad_()
    x = root.clone()
    v1 = x.unbind()
    with self.assertRaises(RuntimeError):
      v1[0].mul_(2)
    v2 = v1[0].narrow(0, 0, 2)
    with self.assertRaises(RuntimeError):
      v2.mul_(2)

  def test_inplace_view_gradcheck(self):
    # gradcheck modifications to views
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)

    def test_fn(root, b):
      x = root.clone()
      x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
      x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
      x.sum().backward()
      return x

    self.runAtenTest((a, b), test_fn)

  def test_inplace_view_makes_base_require_grad(self):
    # in-place modification to view makes base require grad
    a = torch.randn(4, 4, requires_grad=False)
    b = torch.randn(4, 2, requires_grad=True)

    def func(root, b):
      x = root.clone()
      self.assertFalse(x.requires_grad)
      x.narrow(1, 2, 2).mul_(b)
      self.assertTrue(x.requires_grad)
      x.sum().backward()
      self.assertTrue(root.grad is None)
      return b.grad

    self.runAtenTest((a, b), func)

  def test_inplace_view_backprop_view(self):
    # modify view and backprop through view
    xla_device = xm.xla_device()
    a = torch.tensor([2., 5.], device=xla_device, requires_grad=False)
    b = torch.tensor([3.], device=xla_device, requires_grad=True)
    res = a.narrow(0, 1, 1).mul_(b)
    res.sum().backward()
    self.assertEqual(b.grad.tolist(), [5])
    self.assertIsNone(a.grad)

  def test_inplace_view_modify_base(self):
    # Test that an in-place operation on a base that forced it to require
    # grad also forces any previous views to require grad and backprop
    # correctly
    r = torch.ones(1, requires_grad=True)
    x = torch.ones(5)

    def fn(r, x):
      v = x.select(0, 1)
      self.assertFalse(v.requires_grad)
      self.assertIsNone(v.grad_fn)
      x.add_(r)  # v is now dependent on r due to the in-place op on x
      self.assertTrue(v.requires_grad)
      v.sum().backward()
      return r.grad

    self.runAtenTest((r, x), fn)

  def test_inplace_view_python(self):
    # in-place modifications of Python-autograd created view
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)

    class PyAdd(torch.autograd.Function):

      @staticmethod
      def forward(ctx, x, y):
        ctx.mark_dirty(x)
        x.add_(y)
        return x

      @staticmethod
      def backward(ctx, grad):
        return grad, grad

    def func(root, b):
      x = root.clone()
      PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
      PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
      x.sum().backward()
      return root.grad, b.grad

    self.runAtenTest((a, b), func)

  def test_inplace_view_non_contig(self):
    root = torch.ones(
        2, 3, 2, device=xm.xla_device()).select(2, 1).t().requires_grad_(True)
    x = root.clone()
    v1 = x.narrow(0, 0, 1)
    v2 = v1.narrow(1, 1, 1)
    v2.mul_(2)
    x.sum().backward()
    self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

  def test_view_out_computation(self):

    def func(a, b):
      v = a.view(2, 2)
      torch.add(b, 1, out=v)
      return a, v

    a = torch.zeros(4)
    b = torch.ones([2, 2])
    self.runAtenTest((a, b), func)

  def test_multi_view(self):

    def func(x):
      a1, b1 = x.chunk(2)
      a2, b2 = x[0:1], x[1:2]
      a3, b3 = x[0].unsqueeze(0), x[1].unsqueeze(0)
      a4, b4 = x[0, None], x[1, None]
      return a1.squeeze(), b1.squeeze(), a2.squeeze(), b2.squeeze(), a3.squeeze(
      ), b3.squeeze(), a4.squeeze(), b4.squeeze()

    x = torch.randn(size=[2])
    self.runAtenTest(x, func)

  # TODO - upstream behavior has changed and results in expected DestroyXlaTensor
  # counter as of 11/13/2023. Re-enable after reviewing the change.
  # @skipIfFunctionalizationDisabled("metrics differ")
  @unittest.skip
  def test_set(self):
    met.clear_all()

    t1 = torch.zeros(50, device=xm.xla_device())
    t1 += 1
    xm.mark_step()
    self.assertEqual(met.counter_value('DestroyXlaTensor'), 3)

    t2 = torch.zeros(10, device=xm.xla_device())
    self.assertEqual(met.counter_value('DestroyXlaTensor'), 4)

    t1.set_(t2)
    self.assertEqual(met.counter_value('DestroyXlaTensor'), 6)

    # shouldn't crash
    self.assertTrue(torch.allclose(t2.cpu(), torch.zeros(10)))

  @skipIfFunctionalizationDisabled("metrics differ")
  def test_replace_xla_tensor(self):
    met.clear_all()

    t1 = torch.zeros(50, device=xm.xla_device())
    t1 += 1
    xm.mark_step()
    self.assertEqual(met.counter_value('DestroyXlaTensor'), 3)

    t2 = torch.zeros(10, device=xm.xla_device())
    self.assertEqual(met.counter_value('DestroyXlaTensor'), 4)
    torch_xla._XLAC._replace_xla_tensor(t1, t2)
    self.assertEqual(met.counter_value('DestroyXlaTensor'), 5)

    # shouldn't crash
    self.assertTrue(torch.allclose(t2.cpu(), torch.zeros(10)))

  def test_pred_type(self):
    xla_device = xm.xla_device()
    a = torch.rand(4)
    b = torch.rand(4)
    xla_a = a.to(xla_device)
    xla_b = b.to(xla_device)
    c = (a >= 0.25)
    d = (b >= 0.5)
    xla_c = (xla_a >= 0.25)
    xla_d = (xla_b >= 0.5)
    e = torch.cat([a, b], dim=0)
    xla_e = torch.cat([xla_a, xla_b], dim=0)
    f = e.sum().item()
    xla_f = xla_e.sum().item()
    self.assertEqual(f, xla_f)

    # PRED can be automatically promoted in arithmetic and bitwise ops.
    self.runAtenTest(c, lambda x: x + x.byte())
    self.runAtenTest(c, lambda x: x & x.byte())
    self.runAtenTest(c, lambda x: x | x.byte())
    self.runAtenTest(c, lambda x: x ^ x.byte())

  def test_bitwise_and_not(self):
    xla_device = xm.xla_device()
    a = torch.randint(255, (4,), dtype=torch.long)
    xla_a = a.to(xla_device)

    def test_fn(a):
      return a & (~a)

    self.runAtenTest(a, test_fn)

  def test_s_copy_dtype(self):
    xla_device = xm.xla_device()
    a = torch.rand(10).to(xla_device).to(dtype=torch.uint8)
    b = torch.tensor([0, 1, 2, 3]).to(xla_device)
    self.assertEqual(a[b].dtype, torch.uint8)

  def test_slice_zero_sized_dim(self):
    xla_device = xm.xla_device()
    v = torch.randn(2, 3, 4, 5).to(xla_device)
    y = v[:, :, :, 1]
    z = y[:, 1:1, :]
    self.assertEqual(z.size()[1], 0)

  def test_byte_dtype(self):
    xla_device = xm.xla_device()
    x = torch.ByteTensor([0, 1]).to(xla_device)
    y = torch.ByteTensor([0, 1]).to(xla_device)
    z = x + y
    self.assertEqual(z.dtype, torch.uint8)

  def test_frac_negative(self):
    xla_device = xm.xla_device()
    a = torch.tensor(-3.2)
    b = a.frac()
    xla_a = a.to(xla_device)
    xla_b = xla_a.frac()
    self.assertEqual(b, xla_b)

  def test_flip(self):
    device = xm.xla_device()
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device).view(2, 2, 2)
    self.assertEqual(
        torch.tensor([5, 6, 7, 8, 1, 2, 3, 4]).view(2, 2, 2), data.flip(0))
    self.assertEqual(
        torch.tensor([3, 4, 1, 2, 7, 8, 5, 6]).view(2, 2, 2), data.flip(1))
    self.assertEqual(
        torch.tensor([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2), data.flip(2))
    self.assertEqual(
        torch.tensor([7, 8, 5, 6, 3, 4, 1, 2]).view(2, 2, 2), data.flip(0, 1))
    self.assertEqual(
        torch.tensor([8, 7, 6, 5, 4, 3, 2, 1]).view(2, 2, 2),
        data.flip(0, 1, 2))
    # check for wrap dim
    self.assertEqual(
        torch.tensor([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2), data.flip(-1))
    # check for permute
    self.assertEqual(
        torch.tensor([6, 5, 8, 7, 2, 1, 4, 3]).view(2, 2, 2), data.flip(0, 2))
    self.assertEqual(
        torch.tensor([6, 5, 8, 7, 2, 1, 4, 3]).view(2, 2, 2), data.flip(2, 0))

  def test_flip_check_throws(self):
    device = xm.xla_device()
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device).view(2, 2, 2)
    # not allow flip on the same dim more than once
    self.assertRaises(RuntimeError, lambda: data.flip(0, 1, 1))
    # not allow empty list as input
    self.assertRaises(TypeError, lambda: data.flip())
    # not allow size of flip dim > total dims
    self.assertRaises(RuntimeError, lambda: data.flip(0, 1, 2, 3))
    # not allow dim > max dim
    self.assertRaises(RuntimeError, lambda: data.flip(3))

  def test_flip_expand(self):
    device = xm.xla_device()
    data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device).view(2, 2, 2)
    expanded_data = torch.arange(1, 4, device=device).view(3, 1).expand(3, 2)
    transposed_data = torch.arange(
        1, 9, device=device).view(2, 2, 2).transpose(0, 1)
    self.assertEqual(
        torch.tensor([3, 3, 2, 2, 1, 1]).view(3, 2), expanded_data.flip(0))
    self.assertEqual(
        torch.tensor([8, 7, 4, 3, 6, 5, 2, 1]).view(2, 2, 2),
        transposed_data.flip(0, 1, 2))

  def test_flip_shape(self):
    device = xm.xla_device()
    data = torch.randn(2, 3, 4, device=device)
    size = [2, 3, 4]
    test_dims = []
    for i in range(1, 3):
      test_dims += itertools.combinations(range(len(size)), i)
    for ds in test_dims:
      self.assertEqual(size, list(data.flip(ds).size()))

  def test_flip_rectangular(self):
    device = xm.xla_device()
    data = torch.tensor([1, 2, 3, 4, 5, 6]).view(2, 3).to(device)
    flip0_result = torch.tensor([[4, 5, 6], [1, 2, 3]]).to(device)
    flip1_result = torch.tensor([[3, 2, 1], [6, 5, 4]]).to(device)

    self.assertEqual(flip0_result, data.flip(0))
    self.assertEqual(flip1_result, data.flip(1))

  def test_flip_empty_tensor(self):
    device = xm.xla_device()
    data = torch.tensor([])
    self.assertEqual(data, data.flip(0))

  def test_norm_p0(self):
    # p = 0 is equivalent to nonzero
    xla_device = xm.xla_device()
    a = torch.randn(3, 2)
    xla_a = a.to(xla_device)
    norm = a.norm(p=0)
    xla_norm = xla_a.norm(p=0)
    self.assertEqual(norm, xla_norm)

  def test_slice_start_end(self):

    def test_fn(x):
      return x[:, :, -1:0]

    self.runAtenTest(torch.rand(2, 3, 5), test_fn)

  def test_index_bool(self):

    def test_fn(a):
      neg_ones = torch.ones_like(a) * -1
      neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
      a[True] = neg_ones_expanded
      return a

    self.runAtenTest(torch.rand(2, 3), test_fn)

  def test_split_empty_dim(self):

    def test_fn(a):
      return torch.split(a, 2, dim=0)

    self.runAtenTest(torch.rand(0, 1, 3, 0), test_fn)

  def test_pred_and_u8(self):

    def test_fn(a):
      return torch.isfinite(a) & a.ne(0)

    self.runAtenTest(torch.rand(4, 3), test_fn)

  def test_diagonal_scatter_negative_dim(self):

    def test_fn(input, src):
      return torch.diagonal_scatter(input, src, 0, dim1=-1, dim2=0)

    self.runAtenTest([torch.zeros(3, 3), torch.ones(3)], test_fn)

  def test_scatter_add_bool(self):
    xla_device = xm.xla_device()
    a = torch.tensor([[True, True, True, True, True],
                      [True, True, True, True, True]])
    b = torch.zeros(3, 5, dtype=torch.bool)
    index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
    b.scatter_add_(0, index, a)
    xla_a = a.to(xla_device)
    xla_b = b.to(xla_device)
    xla_index = index.to(xla_device)
    xla_b.scatter_add_(0, xla_index, xla_a)
    self.assertEqual(b, xla_b)

  def test_squeeze_nonzero(self):

    def test_fn(a):
      level = 1
      return torch.nonzero(a == level).squeeze(1)

    self.runAtenTest(torch.tensor([3, 1]), test_fn)

  def test_expand_default(self):

    def test_fn(a):
      return a.expand((1, 1, -1, -1))

    self.runAtenTest(torch.zeros([4, 4]), test_fn)

  def test_stack_pred(self):

    def test_fn(a):
      i0, j0, k0 = a[:-1].t()
      i1, j1, k1 = a[1:].t()
      i_ok = i1 >= i0
      j_ok = (j1 >= j0) | (i1 > i0)
      return torch.stack([i_ok, j_ok], dim=1)

    self.runAtenTest(torch.randint(3, (7, 3)), test_fn)

  def test_reduction_0dim(self):
    self.runAtenTest(torch.rand(2, 0, 4).bool(), lambda x: torch.all(x))
    self.runAtenTest(torch.rand(2, 0, 4).bool(), lambda x: torch.any(x))
    self.runAtenTest(torch.rand(2, 0, 4), lambda x: torch.sum(x))
    self.runAtenTest(torch.rand(2, 0, 4), lambda x: torch.mean(x))
    self.runAtenTest(torch.rand(2, 0, 4), lambda x: torch.prod(x))
    # min & max throws
    xla_device = xm.xla_device()
    a = torch.rand(2, 0, 4)
    xla_a = a.to(xla_device)
    self.assertRaises(IndexError, lambda: torch.max(a, dim=1))
    self.assertRaises(RuntimeError, lambda: torch.max(a))
    self.assertRaises(IndexError, lambda: torch.min(a, dim=1))
    self.assertRaises(RuntimeError, lambda: torch.min(a))
    self.assertRaises(RuntimeError, lambda: torch.max(xla_a, dim=1))
    self.assertRaises(RuntimeError, lambda: torch.max(xla_a))
    self.assertRaises(RuntimeError, lambda: torch.min(xla_a, dim=1))
    self.assertRaises(RuntimeError, lambda: torch.min(xla_a))

  def test_reduction_unordered_dim(self):
    self.runAtenTest(
        torch.rand(4, 3, 4, 2),
        lambda x: torch.mean(x, (-1, -3, -2), keepdim=True))

  def test_index_select_0dim(self):

    def test_fn(s, i):
      return torch.index_select(s, 0, i)

    self.runAtenTest(
        [torch.randn(0, 1, 2, 0),
         torch.tensor([], dtype=torch.long)], test_fn)

  def test_scatter_add_small_target(self):

    def test_fn(t, s, i):
      t.scatter_add_(1, i, s)
      return t

    self.runAtenTest(
        [torch.randn(2, 4),
         torch.randn(2, 8),
         torch.randint(0, 4, (2, 8))], test_fn)

  def test_diagonal_write(self):

    def test_fn(t):
      d = torch.diagonal(t, offset=1)
      d[1] += 0.904
      return t

    self.runAtenTest([torch.randn(5, 8)], test_fn)

  def test_diagonal_write_transposed(self):

    def test_fn(t):
      d = torch.diagonal(t, offset=-1, dim1=1, dim2=0)
      d[1] += 0.904
      return t

    self.runAtenTest([torch.randn(5, 8)], test_fn)

  def test_diagonal_write_transposed_r3(self):

    def test_fn(t):
      d = torch.diagonal(t, offset=1, dim1=2, dim2=0)
      d[1] += 0.904
      return t

    self.runAtenTest([torch.randn(5, 8, 7)], test_fn)

  def test_writeable_tensors_updates(self):

    def test_fn(s, i):
      out = torch.zeros(2, 4, device=s.device)
      return torch.index_select(s, 0, i, out=out)

    self.runAtenTest(
        [torch.randn(3, 4),
         torch.tensor([2, 1], dtype=torch.long)], test_fn)

  def test_index_select_out(self):

    def test_fn(s, i):
      out = torch.randn(5 * 4 * 5, device=s.device)
      return torch.index_select(s, 0, i, out=out.view(5, 4, 5)), out

    self.runAtenTest(
        [torch.randn(3, 4, 5),
         torch.tensor([2, 1, 0, 1, 2], dtype=torch.long)], test_fn)

  def test_pred_one_hot(self):

    def test_fn(t, c):
      s = (t[:, None] != c[None, :]).long()
      return F.one_hot(s, num_classes=2)

    token_type_ids = torch.randint(
        1, 5, (
            128,
            32,
        ), dtype=torch.int64)
    cat_ids = torch.randint(
        1, 5, (
            128,
            32,
        ), dtype=torch.int64)
    self.runAtenTest([token_type_ids, cat_ids], test_fn)

  def test_one_hot_no_fallback(self):

    def test_fn(t):
      met.clear_all()
      res = F.one_hot(t, num_classes=5)
      # make sure there is no graph break
      assert 'aten::' not in met.short_metrics_report()
      return res

    t1 = torch.arange(0, 5) % 3

    self.runAtenTest([t1], test_fn)

  @skipIfFunctionalizationEnabled("views do not exist")
  def test_save_view_alias_check(self):

    class Nested(object):

      def __init__(self, x, y):
        self.x = x
        self.y = y

    def check(device):
      a = torch.rand(16, device=device)
      b = a[:10]
      c = a[6:]
      self.assertRaises(RuntimeError, lambda: xm.check_view_sharing([b, c]))

      nested = Nested(b, c)
      self.assertRaises(RuntimeError, lambda: xm.check_view_sharing(nested))

      d = a
      xm.check_view_sharing([a, d])

    check(xm.xla_device())
    check(torch.device('cpu'))

  def test_save(self):
    xla_device = xm.xla_device()
    x = torch.randn(5, device=xla_device)
    with tempfile.NamedTemporaryFile() as tf:
      torch.save(x, tf)
      x_loaded = torch.load(tf.name)
      self.assertEqual(x, x_loaded)

  def test_save_bf16(self):
    xla_device = xm.xla_device()
    x = torch.randn(5, dtype=torch.bfloat16, device=xla_device)
    with tempfile.NamedTemporaryFile() as tf:
      torch.save(x, tf)
      x_loaded = torch.load(tf.name)
      self.assertEqual(x, x_loaded)

  def test_save_tuple(self):
    xla_device = xm.xla_device()
    x = torch.randn(5, device=xla_device)
    number = 3
    with tempfile.NamedTemporaryFile() as tf:
      torch.save((x, number), tf)
      x_loaded, number_loaded = torch.load(tf.name)
      self.assertEqual(x, x_loaded)
      self.assertEqual(number, number_loaded)

  def test_save_api(self):
    xla_device = xm.xla_device()
    model = XlaMNIST().to(xla_device)
    with tempfile.NamedTemporaryFile() as tf:
      xm.save(model.state_dict(), tf)
      state_dict = torch.load(tf.name)
    cpu_model = XlaMNIST()
    cpu_model.load_state_dict(state_dict)
    loaded_model = cpu_model.to(xla_device)
    self.assertEqual(model.state_dict(), loaded_model.state_dict())

  def test_serialization_api(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'data.pt')
      xla_device = xm.xla_device()
      model = XlaMNIST().to(xla_device)
      xser.save(model.state_dict(), path)
      state_dict = xser.load(path)
      cpu_model = XlaMNIST()
      cpu_model.load_state_dict(state_dict)
      loaded_model = cpu_model.to(xla_device)
      self.assertEqual(model.state_dict(), loaded_model.state_dict())

  def test_deepcopy(self):
    xla_device = xm.xla_device()
    x = torch.rand(5, device=xla_device)
    x0 = x[0]
    y = copy.deepcopy(x)
    self.assertEqual(x, y)
    y[0] = 1
    # Make sure x doesn't change with y.
    self.assertEqual(x[0], x0)

  def test_print(self):
    xla_device = xm.xla_device()
    x = torch.tensor([5], device=xla_device)
    expected_str = 'tensor([5], device=\'' + str(xla_device) + '\')'
    self.assertEqual(str(x), expected_str)

    def test_type_promotion_issue_1929(self):

      def test_fn(m, x):
        x = torch.matmul(x, m)
        x *= 0.5
        loss = x.sum()
        loss.backward()
        return m.grad

      self.runAtenTest([
          torch.rand(5, 10, requires_grad=True),
          torch.rand(8, 5, requires_grad=True)
      ], test_fn)

  def test_as_strided_r1(self):

    def test_fn(r):
      return torch.as_strided(r, (5, 3), (3, 1))

    self.runAtenTest([torch.arange(15, dtype=torch.int32)], test_fn)

  def test_as_strided_r1_t(self):

    def test_fn(r):
      return torch.as_strided(r, (5, 3), (1, 5))

    self.runAtenTest([torch.arange(15, dtype=torch.int32)], test_fn)

  def test_as_strided_r1_t_off(self):

    def test_fn(r):
      return torch.as_strided(r, (5, 2, 3), (1, 15, 5), 5)

    self.runAtenTest([torch.arange(35, dtype=torch.int32)], test_fn)

  def test_as_strided_r2_t_update(self):

    def test_fn(r):
      a = torch.as_strided(r, (5, 2, 3), (1, 15, 5))
      a[1, 0, 2] = -1
      return a

    self.runAtenTest([torch.arange(30, dtype=torch.int32)], test_fn)

  def test_as_strided_r1_slice(self):

    def test_fn(r):
      v = r.view(5, 3)
      return torch.as_strided(v, (4, 3), (3, 1), 3)

    self.runAtenTest([torch.arange(15, dtype=torch.int32)], test_fn)

  def test_as_strided_r1_t_slice(self):

    def test_fn(r):
      v = r.view(5, 3)
      return torch.as_strided(v, (5, 2), (1, 5), 5)

    self.runAtenTest([torch.arange(15, dtype=torch.int32)], test_fn)

  def test_as_strided_r1_dim1(self):

    def test_fn(r):
      return torch.as_strided(r, (2, 1, 3, 4, 6), (12, 12, 4, 1, 24))

    self.runAtenTest([torch.arange(144, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_gap(self):

    def test_fn(r):
      return torch.as_strided(r, (4, 4), (8, 1))

    self.runAtenTest([torch.arange(28, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_gap_no_unit_stride(self):

    def test_fn(r):
      return torch.as_strided(r, (4, 4), (8, 2))

    self.runAtenTest([torch.arange(31, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_overlap(self):

    def test_fn(r):
      return torch.as_strided(r, (4, 4), (2, 1))

    self.runAtenTest([torch.arange(10, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_overlap_and_gap(self):

    def test_fn(r):
      return torch.as_strided(r, (4, 4), (4, 2))

    self.runAtenTest([torch.arange(19, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_overlap_zero_stride(self):

    def test_fn(r):
      return torch.as_strided(r, (4, 4), (0, 1))

    self.runAtenTest([torch.arange(19, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_gap_no_unit_stride(self):

    def test_fn(r):
      x = r.view(8, 4)
      return torch.as_strided(r, (4, 4), (6, 2))

    self.runAtenTest([torch.arange(32, dtype=torch.int32)], test_fn)

  @skipIfFunctionalizationDisabled("arbitrary as_strided unsupported")
  def test_as_strided_with_empty_args(self):

    def test_fn(r):
      return torch.as_strided(r, tuple(), tuple())

    self.runAtenTest([torch.arange(32, dtype=torch.int32)], test_fn)

  def test_basic_bfloat16(self):

    def test_fn(s):
      return s * torch.tensor(2.3, dtype=torch.float32)

    self.runAtenTest([torch.ones(2, 2, dtype=torch.bfloat16)], test_fn)

  def test_float32_bfloat16_cast(self):

    def test_fn(s, t):
      s = s.to(torch.bfloat16)
      t = t.to(torch.bfloat16)
      v = s * t
      return v.to(torch.float32)

    self.runAtenTest([torch.ones(2, 2), torch.randn(2, 2)], test_fn)

  def test_bfloat16_float32_cast(self):

    def test_fn(s, t):
      s = s.to(torch.float32)
      t = t.to(torch.float32)
      v = s * t
      return v.to(torch.bfloat16)

    self.runAtenTest([
        torch.ones(2, 2).to(torch.bfloat16),
        torch.randn(2, 2).to(torch.bfloat16)
    ], test_fn)

  def test_inplace_copy_different_sizes(self):

    def test_fn(t, r):
      t.copy_(r)
      return t

    self.runAtenTest([torch.rand(2, 4), torch.zeros(2, 1)], test_fn)

  def test_spooky_ailing(self):

    def test_fn(m):
      a = torch.pow(3, m[4])
      r1 = torch.empty_like(a).zero_()
      for i in range(r1.size(0)):
        r1[i] = math.pow(3, m[4][i])
      s = str(r1)

      b = torch.pow(3, m[:, 4])
      r2 = torch.empty_like(b).zero_()
      for i in range(r2.size(0)):
        r2[i] = math.pow(3, m[i][4])
      return r1, r2

    self.runAtenTest([torch.randint(1, 4, (7, 7), dtype=torch.uint8)], test_fn)

  def test_too_many_parameter(self):

    def test_fn(t):
      # TPU can handle ~3500 parameters on v3 without parameter tupling.
      for i in range(4000):
        t += torch.tensor(i, dtype=torch.float, device=t.device)
      return t

    self.runAtenTest([torch.tensor(20.0)], test_fn)

  def test_view_and_copy_(self):
    xla_device = xm.xla_device()
    x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], device='cpu')
    y = torch.tensor([0, 0, 0, 0, 0, 0], device=xla_device)
    y[::2].copy_(x[::2])
    self.assertEqual(y, [1, 0, 3, 0, 5, 0])

  def test_view_and_multi_mark_step(self):
    xla_device = xm.xla_device()
    t1 = torch.zeros(100, device=xla_device)
    t1[10] = 113
    xm.mark_step()
    t1[12] = 1123
    xm.mark_step()
    self.assertNotIn('update_slice',
                     torch_xla._XLAC._get_xla_tensors_text([t1]))

  def test_binaryop_order(self):
    xla_device = xm.xla_device()
    x = torch.rand(5, device=xla_device)
    y = torch.rand(5)
    self.assertEqual(x + y, y + x)

  # Since in eager mode the tensor would be materialized and hence _get_xla_tensors_text would not show the prim::Constant node.
  @skipOnEagerDebug
  def test_pow_constant(self):
    t1 = torch.pow(torch.tensor([2.0, 3.0], device=xm.xla_device()), 5)
    hlo_text = torch_xla._XLAC._get_xla_tensors_text([t1])
    const_hlo = hlo_text.split('\n')[1]
    assert 'prim::Constant' in const_hlo
    assert 'xla::device_data' not in const_hlo

  def test_emb_bf16(self):
    xla_device = xm.xla_device()
    index = torch.ones(1, dtype=torch.long, device=xla_device)
    emb = torch.nn.Embedding(1024, 128, device=xla_device)
    emb = emb.to(torch.bfloat16)
    emb_out = emb(index)
    assert emb_out.dtype == torch.bfloat16

  def test_embedding_int_indices(self):
    model = torch.nn.Embedding(1024, 10)

    # 1 and 2-dimensional tensors.
    # They have different execution paths.
    for shape in ((5,), (2, 5)):

      def test_on_device(device):
        m = copy.deepcopy(model).to(device)
        index = torch.ones(shape, dtype=torch.int, device=device)
        return m(index)

      out = test_on_device("cpu")
      out_x = test_on_device(xm.xla_device())
      self.assertEqual(out, out_x.cpu())

  def test_transpose_1d(self):

    def test_fn(t1):
      return t1.t()

    self.runAtenTest([torch.arange(15, dtype=torch.int32)], test_fn)

  def test_transpose_1d_inplace(self):

    def test_fn(t1):
      return t1.t_()

    self.runAtenTest([torch.arange(15, dtype=torch.int32)], test_fn)

  def test_sigmoid_bounds(self):
    torch.manual_seed(0)
    xla_device = xm.xla_device()
    for _ in range(100):
      x = torch.rand(1000).to(xla_device)
      lower_bound = torch.sigmoid(x * (-100.0))
      upper_bound = torch.sigmoid(x * (100.0))
      assert torch.all(lower_bound >= 0.0)
      assert torch.all(upper_bound <= 1.0)

  def test_manual_seed(self):
    device = torch_xla.device()
    torch_xla.manual_seed(12345)
    t1 = torch.randn(5, 5, device=device)
    torch_xla.manual_seed(12345)
    t2 = torch.randn(5, 5, device=device)
    self.assertTrue(torch.allclose(t1.cpu(), t2.cpu()))

  def test_cached_addcdiv(self):
    xla_device = xm.xla_device()
    met.clear_all()

    t1 = torch.randn(1, 3).to(xla_device)
    t2 = torch.randn(1, 3).to(xla_device)
    t3 = torch.randn(1, 3).to(xla_device)
    t1.addcdiv_(t2, t3, value=0.1)
    xm.mark_step()
    self.assertEqual(met.metric_data("TransferToDeviceTime")[0], 4)

    # The following two scalars shouldn't trigger TransferToDeviceTime.
    t1.addcdiv_(t2, t3, value=0.1)
    t1.addcdiv_(t2, t3, value=0.1)
    xm.mark_step()
    self.assertEqual(met.metric_data("TransferToDeviceTime")[0], 4)

  @skipOnEagerDebug
  def test_print_executation(self):
    xla_device = xm.xla_device()
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()

    # case 1 mark_step
    t1 = torch.randn(1, 4, device=xla_device)
    xm.mark_step()
    xm.wait_device_ops()
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)
    for _ in range(3):
      print(t1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)
    self.assertIn('xla::device_data',
                  torch_xla._XLAC._get_xla_tensors_text([t1]))

    # case 2 no mark_step, directly print
    met.clear_all()
    t1 = torch.randn(1, 4, device=xla_device)
    for _ in range(3):
      print(t1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)
    self.assertIn('xla::device_data',
                  torch_xla._XLAC._get_xla_tensors_text([t1]))

    # case 2 no mark_step, print with .cpu
    met.clear_all()
    t1 = torch.randn(1, 4, device=xla_device)
    for _ in range(3):
      print(t1.cpu())
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)
    self.assertIn('xla::device_data',
                  torch_xla._XLAC._get_xla_tensors_text([t1]))

  def test_index_types(self):

    def test_fn(*indices):
      x = torch.arange(10).to(indices[0].device)
      return [x[idx] for idx in indices]

    self.runAtenTest([
        torch.randint(0, 1, size=(10,), dtype=dtype)
        for dtype in (torch.long, torch.int32, torch.bool)
    ], test_fn)

  def test_native_dropout_backward(self):

    def test_fn(input):
      dropped = torch.native_dropout(input, 0.5, train=True)
      loss = dropped[0] + 0.5
      loss.mean().backward()
      return dropped[1].cpu(), input.grad.cpu()

    met.clear_all()
    xla_device = xm.xla_device()
    input_cpu = torch.randn(7, 7, requires_grad=True)
    input_xla = torch.randn(7, 7, device=xla_device, requires_grad=True)
    mask_cpu, grad_cpu = test_fn(input_cpu)
    mask_xla, grad_xla = test_fn(input_xla)
    # dropout is random, hence we construct the expected grad_xla by mask_xla
    # and gradient_cpu.
    grad_cpu_single = grad_cpu[mask_cpu][0]
    torch.allclose(
        grad_cpu_single * mask_xla.to(torch.float), grad_xla, rtol=1e-03)

    self.assertIn("xla::native_dropout_backward", met.counter_names())
    self.assertNotIn("aten::native_dropout_backward", met.counter_names())

  def test_conv2d_backward(self):
    # Somehow eager cpu produces different results than us, and
    # therefore we can't compare eager and xla.
    conv = nn.Conv2d(1, 1, kernel_size=1).to('xla')
    input = torch.tensor([[[[2077.0]]]]).to('xla')

    output = conv(input)
    loss = torch.sum(output)
    loss.backward()
    self.assertTrue(
        torch.allclose(conv.weight.grad.cpu(), torch.tensor([[[[2077.0]]]])))

  @skipOnTpu  # fail with precision issue on TPU
  def test_patched_linear_3D(self):
    linear_cpu = nn.Linear(2, 4, bias=False)
    input_cpu = torch.randn(4, 3, 2, requires_grad=True)
    input_cpu.retain_grad()
    output_cpu = linear_cpu(input_cpu)

    # It looks like nn.Module.to is in-place.
    linear = copy.deepcopy(linear_cpu).to('xla')
    apply_xla_patch_to_nn_linear(linear, xs.xla_patched_nn_linear_forward)
    input = copy.deepcopy(input_cpu).to('xla')
    input.retain_grad()
    output = linear(input)

    # Make sure that we don't have any reshapes in the patched linear.
    hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
    self.assertNotIn("reshape", hlo)

    # Make sure the forward result is correct.
    self.assertTrue(torch.allclose(output.cpu(), output_cpu))

    # Now work on the backward.
    linear_cpu.weight.retain_grad()
    loss_cpu = output_cpu.sum()
    loss_cpu.backward()

    loss = output.sum()
    loss.backward()

    self.assertTrue(
        torch.allclose(linear.weight.grad.cpu(), linear_cpu.weight.grad))
    self.assertTrue(torch.allclose(input.grad.cpu(), input_cpu.grad))

  @skipOnTpu  # fail with precision issue on TPU
  def test_patched_linear_3D_bias(self):
    linear_cpu = nn.Linear(2, 4)
    input_cpu = torch.randn(4, 3, 2)
    output_cpu = linear_cpu(input_cpu)

    # It looks like nn.Module.to is in-place.
    linear = copy.deepcopy(linear_cpu).to('xla')
    apply_xla_patch_to_nn_linear(linear, xs.xla_patched_nn_linear_forward)
    input = copy.deepcopy(input_cpu).to('xla')
    output = linear(input)

    # We will have some reshapes on the bias. So skip the check here.
    # Make sure the forward result is correct.
    self.assertTrue(torch.allclose(output.cpu(), output_cpu))

    # Now work on the backward.
    linear_cpu.weight.retain_grad()
    loss_cpu = output_cpu.sum()
    loss_cpu.backward()

    loss = output.sum()
    loss.backward()

    self.assertTrue(
        torch.allclose(linear.bias.grad.cpu(), linear_cpu.bias.grad))

  @skipOnTpu  # fail with precision issue on TPU
  def test_patched_linear_2D_bias(self):
    linear_cpu = nn.Linear(2, 4)
    input_cpu = torch.randn(4, 2, requires_grad=True)
    input_cpu.retain_grad()
    output_cpu = linear_cpu(input_cpu)

    # It looks like nn.Module.to is in-place.
    linear = copy.deepcopy(linear_cpu).to('xla')
    apply_xla_patch_to_nn_linear(linear, xs.xla_patched_nn_linear_forward)
    input = copy.deepcopy(input_cpu).to('xla')
    input.retain_grad()
    output = linear(input)

    # Make sure the forward result is correct.
    self.assertTrue(torch.allclose(output.cpu(), output_cpu))

    # Now work on the backward.
    linear_cpu.weight.retain_grad()
    loss_cpu = output_cpu.sum()
    loss_cpu.backward()

    loss = output.sum()
    loss.backward()

    self.assertTrue(
        torch.allclose(linear.weight.grad.cpu(), linear_cpu.weight.grad))
    self.assertTrue(torch.allclose(input.grad.cpu(), input_cpu.grad))
    self.assertTrue(
        torch.allclose(linear.bias.grad.cpu(), linear_cpu.bias.grad))

  @skipOnTpu  # fail with precision issue on TPU
  def test_patched_linear_1D_bias(self):
    linear_cpu = nn.Linear(2, 4)
    input_cpu = torch.randn(2, requires_grad=True)
    input_cpu.retain_grad()
    output_cpu = linear_cpu(input_cpu)

    # It looks like nn.Module.to is in-place.
    linear = copy.deepcopy(linear_cpu).to('xla')
    apply_xla_patch_to_nn_linear(linear, xs.xla_patched_nn_linear_forward)
    input = copy.deepcopy(input_cpu).to('xla')
    input.retain_grad()
    output = linear(input)

    # Make sure the forward result is correct.
    self.assertTrue(torch.allclose(output.cpu(), output_cpu))

    # Now work on the backward.
    linear_cpu.weight.retain_grad()
    loss_cpu = output_cpu.sum()
    loss_cpu.backward()

    loss = output.sum()
    loss.backward()

    self.assertTrue(
        torch.allclose(linear.weight.grad.cpu(), linear_cpu.weight.grad))
    self.assertTrue(torch.allclose(input.grad.cpu(), input_cpu.grad))
    self.assertTrue(
        torch.allclose(linear.bias.grad.cpu(), linear_cpu.bias.grad))

  def test_pow_dtype_promotion(self):

    def test(dtype):

      def foo(x):
        return torch.pow(x, 3.0)

      x = torch.arange(10).to(dtype)
      r = foo(x)

      device = xm.xla_device()
      Xx = x.to(device)
      Xr = foo(Xx)

      self.assertEqual(r, Xr.cpu())

    test_dtypes = [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.cfloat,
    ]

    if not _is_on_tpu():
      test_dtypes += [
          torch.cdouble,
      ]

    for dtype in test_dtypes:
      test(dtype)

  def test_trilinear_interpolate(self):

    def func(input_volume):
      output_size = (32, 64, 64)
      return F.interpolate(
          input_volume, size=output_size, mode='trilinear', align_corners=False)

    device = torch_xla.device()
    input_volume = torch.randn(1, 3, 16, 32, 32).to(device)
    met.clear_all()
    self.runAtenTest((input_volume), func)
    assert len(torch_xla._XLAC._get_executed_fallback_ops()) == 0

  def test_gelu_backward_different_types(self):

    def foo(grad, inp):
      return torch.ops.aten.gelu_backward.default(grad, inp)

    grad = torch.rand(10, 10, dtype=torch.bfloat16)
    inp = torch.rand(10, 10)

    Xgrad = grad.to(xm.xla_device())
    Xinp = inp.to(xm.xla_device())

    r = foo(grad, inp)
    Xr = foo(Xgrad, Xinp)

    self.assertEqual(r, Xr.cpu())

  def test_clip_grad_norm_(self):

    def foo(t):
      torch.nn.utils.clip_grad_norm_(t, 1.0)

    t = torch.rand(10, 10, requires_grad=True, dtype=torch.bfloat16)
    t.retain_grad()
    t.grad = torch.rand(10, 10, dtype=torch.bfloat16)
    xt = t.to(xm.xla_device())
    xt.grad = t.grad.to(xm.xla_device(), dtype=torch.bfloat16)

    foo(t)
    foo(xt)

    self.assertEqual(xt.grad.dtype, torch.bfloat16)
    self.assertEqual(t.grad, xt.grad.cpu())

  def test_clip_grad_norm_zero(self):
    t = torch.rand(10, 10, dtype=torch.bfloat16)
    xt = t.to(xm.xla_device())
    result = torch.nn.utils.clip_grad_norm_(xt, 1.0)
    self.assertEqual(result.device.type, 'xla')
    self.assertTrue(torch.allclose(result.cpu(), torch.tensor(0.)))

  def test_stack_different_types(self):

    def foo(t0, t1):
      return torch.stack([t0, t1])

    t0 = torch.rand(10, 10, dtype=torch.bfloat16)
    t1 = torch.rand(10, 10)

    Xt0 = t0.to(xm.xla_device())
    Xt1 = t1.to(xm.xla_device())

    r = foo(t0, t1)
    Xr = foo(Xt0, Xt1)

    self.assertEqual(r, Xr.cpu())

  def test_index_zero_tensor_by_zero_tensor(self):

    # Test if simple one-tensor indexing works.
    # Should return a non-permuted tensor.
    def f1(x, i):
      return x[i]

    # Test if scattered two-tensor indexing works.
    # Should return a permuted tensor, with indexed dimensions first.
    def f2(x, i0, i1):
      return x[:, i0, :, i1]

    cases = {
        f1: [
            ((0,), (0,)),
            ((0, 10), (0, 5, 5)),
            ((0, 3, 3), (5, 5, 0)),
        ],
        f2: [
            ((10, 0, 10, 10), (5, 0, 5), (5, 1, 1)),
            ((0, 0, 10, 0), (5, 5, 0), (5, 5, 1)),
        ]
    }

    def make_tensor(shape):
      return torch.rand(shape)

    def make_index(shape):
      return torch.randint(0, 100, shape, dtype=torch.long)

    def test(f, xshape, ishapes):
      x = make_tensor(xshape)
      ilist = [make_index(s) for s in ishapes]

      Xx = x.to(xm.xla_device())
      Xilist = [i.to(xm.xla_device()) for i in ilist]

      out = f(x, *ilist)
      Xout = f(Xx, *Xilist)

      self.assertEqual(out, Xout.cpu())

    for xshape, ishape in cases[f1]:
      test(f1, xshape, (ishape,))

    for xshape, i0shape, i1shape in cases[f2]:
      test(f2, xshape, (i0shape, i1shape))

  def test_inplace_mul_scalar_different_dtype(self):
    # This tests whether the returned output data-type agrees on PyTorch
    # and XLA sides.
    #
    # Technical details: even though we were computing the common data-type
    # inside PyTorch/XLA XLANativeFunctions::mul function, we were using it
    # just for telling PyTorch what the output data-type would be, i.e. creating
    # an IR node of that data-type). Meanwhile, in the XLA side of things,
    # it would just promote the tensors using other data-type promotion rules.
    #
    # In summary, given the expressions below, the problem this test covers is:
    #
    #   >>> t = torch.rand(10, dtype=torch.half)
    #   >>> s = torch.tensor(5, dtype=torch.double)
    #   >>> out = t.mul_(s)
    #
    #   out.dtype is torch.float16, but its underlying XLA type (xla::Shape's
    #   element_type) is F64
    #
    # See: https://github.com/pytorch/xla/issues/7084

    def fn(inp, s):
      return inp.mul_(s)

    inp = torch.rand(10, dtype=torch.half)
    s = torch.tensor(7, dtype=torch.double)

    Xinp = inp.to(xm.xla_device())
    Xs = s.to(xm.xla_device())

    out = fn(inp, s)
    Xout = fn(Xinp, Xs)

    self.assertEqual(out, Xout.cpu())
    self.assertEqual("f16", torch_xla._XLAC._get_xla_tensor_shape_type(Xout))

  # We skip TPU for 2 reasons:
  #   1. upsample_bilinear on f64 tensors doesn't work on TPUs
  #   2. This issue only affects non-TPU and non-Neuron devices (i.e. there's
  #      a short-circuit for both devices that don't go through the bug path)
  @skipOnTpu
  def test_upsample_bilinear_double(self):
    # Originally, the upsample_bilinear implementation (in resize_ops.cpp)
    # was copied from TF. The computation was done intentionally on F32 and
    # not cast back[1]. However, that didn't reflect in the returned tensor.
    # Basically, what would happen is:
    #
    # 1. A tensor of data-type other than F32 is created:
    #    > a = torch.rand(..., dtype=torch.double)
    #
    # 2. Call upsample_bilinear on it
    #    > r = torch.nn.functional.upsample_bilinear(a, scale_factor=2)
    #
    # 3. The result's data-type would show as torch.float64, but its inner
    #    HLO representation would be actually F32.
    #
    #     - It would rarely surface as an error, since we do data-type
    #       promotion at the HLO level.
    #
    #     - When this result is the argument of a new HLO function, XLA
    #       would actually expect a F16 tensor, since its torch.Tensor
    #       data-type "is" torch.float16. However, since the actual HLO
    #       data-type is F32, XLA raises an error.
    #
    # See more details at [2].
    #
    # [1]: https://github.com/tensorflow/tensorflow/commit/f8b35e00afe09c8606bcb0441a51be8bd38168d2
    # [2]: https://github.com/pytorch/xla/issues/7095

    def foo(x, is_xla=False):
      # Compute upsample_bilinear.
      r = torch.nn.functional.upsample_bilinear(x, scale_factor=2)

      if is_xla:
        # Mark the end of the HLO graph.
        xm.mark_step()

      # Start a new HLO graph using the upsample_bilinear result as
      # one of its arguments.
      return r + 5

    inp = torch.rand(1, 3, 10, 10, dtype=torch.double)
    Xinp = inp.to(xm.xla_device())

    out = foo(inp)
    Xout = foo(Xinp, is_xla=True)

    self.assertEqual(out, Xout.cpu())

  def test_embedding_bag_backward_fallback(self):
    # Tests whether EmbeddingBag backward function works and computes the expected results.
    #
    # EmbeddingBag has a 'sparse' flag which dictates what will be the layout of the grad
    # returned by its backward function. Unfortunately, PyTorch/XLA doesn't support sparse
    # tensors, yet. Therefore, as a work-around, we fallback to the dense backward function.
    #
    # This test tests whether we correctly compute the backward for sparse=True and
    # sparse=False, making sure that we did not introduce any regressions.

    # Run EmbeddingBag forward and backwards.
    # Return the forward result + the computed weight grad.
    def fn(indices, weight, **kwargs):
      out = F.embedding_bag(indices, weight, **kwargs)
      out.sum().backward()
      return out, weight.grad

    # Clone a tensor, and maybe move it to a different device.
    def clone_and_maybe_move(tensor, device=None):
      fresh = tensor
      # Maybe move to the specified device.
      if device is not None:
        fresh = fresh.to(device)
      # Clone if not cloned already by the previous device move.
      if fresh.device == tensor.device and fresh.data_ptr() == tensor.data_ptr(
      ):
        fresh = tensor.clone()
      # Make this tensor a leaf tensor by detaching and reseting its
      # requires_grad property.
      fresh = fresh.detach()
      fresh.requires_grad_(tensor.requires_grad)
      return fresh

    EMBEDDINGS = 10
    VECTOR_DIM = 5
    N = 5

    kwargs = {
        "indices": torch.randint(0, EMBEDDINGS, (N,)),
        "weight": torch.randn((EMBEDDINGS, VECTOR_DIM), requires_grad=True),
        "offsets": torch.tensor([0, 3], dtype=torch.long),
    }

    # Test all combinations of sparse + mode.
    for sparse, mode in itertools.product((False, True),
                                          ("sum", "mean", "max")):
      # According to nn.functional.embedding_bag PyTorch documentation, not supported.
      if sparse and mode == "max":
        continue

      extra_kwargs = {
          "mode": mode,
          "sparse": sparse,
      }

      with self.subTest(sparse=sparse, mode=mode):
        kwargs_ = {k: clone_and_maybe_move(v) for k, v in kwargs.items()}
        xla_kwargs = {
            k: clone_and_maybe_move(v, device=xm.xla_device())
            for k, v in kwargs.items()
        }

        expected_out, expected_grad = fn(**kwargs_, **extra_kwargs)
        actual_out, actual_grad = fn(**xla_kwargs, **extra_kwargs)

        # PyTorch/XLA doesn't support sparse tensors.
        # We explicitly fallback to the dense backward function whenever sparse=True.
        # Therefore, we have to convert the expected grad to dense, so that we can
        # compare the actual numbers.
        if sparse:
          self.assertTrue(expected_grad.is_sparse)
          self.assertFalse(actual_grad.is_sparse)
          expected_grad = expected_grad.to_dense()

        self.assertEqual(actual_out, expected_out)
        self.assertEqual(actual_grad, expected_grad)

  def test_amp_norm_append_dtype(self):
    # Tests whether the returned tensor is actually of the specified dtype.
    #
    # The operation norm.ScalarOpt_dim_dtype is actually called when using AMPl. It
    # is redirected from norm.ScalarOpt_dim by appending kFloat data-type as its last
    # argument.

    def foo(x: torch.Tensor) -> torch.Tensor:
      return torch.ops.aten.norm.ScalarOpt_dim_dtype(
          x, p=2, dim=1, keepdim=True, dtype=torch.float32)

    input = torch.rand((10, 10), dtype=torch.float16)
    out = foo(input)

    in_xla = input.to(xm.xla_device())
    out_xla = foo(in_xla)

    self.assertEqual(out.dtype, out_xla.dtype)
    self.assertEqual(out.cpu(), out_xla.cpu(), prec=1e-4)


class MNISTComparator(nn.Module):

  def __init__(self):
    super(MNISTComparator, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    mc.save(None, x)
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    mc.save('layer1', x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    mc.save('layer2', x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    mc.save('relu', x)
    x = self.fc2(x)
    x = F.log_softmax(x, dim=1)
    mc.save('result', x)
    return x


class TestModelComparator(test_utils.XlaTestCase):

  def test(self):
    SEED = 42

    xla_device = xm.xla_device()
    x = _gen_tensor(8, 1, 28, 28)
    xla_x = x.to(xla_device)

    test_utils._set_rng_seed(SEED)
    model = MNISTComparator()
    save_dir1 = xu.TmpFolder()
    mc.configure(save_dir1.name)
    model(x)

    save_dir2 = xu.TmpFolder()
    mc.configure(save_dir2.name)
    test_utils._set_rng_seed(SEED)
    xla_model = MNISTComparator().to(xla_device)
    xla_model(xla_x)

    report = mc.compare(save_dir1.name, save_dir2.name, rtol=1e-03, atol=1e-03)
    if report:
      print(report)
    self.assertEqual(len(report), 0)


class TestWaitDeviceOps(test_utils.XlaTestCase):

  def test_wait_device_ops(self):
    xm.xla_device()
    value = torch.randn(10000, 10000, device=xm.xla_device())
    val_list = []
    val_mean_list = []
    met.clear_all()
    for _ in range(5):
      new_val = value * torch.randn(10000, 10000, device=xm.xla_device())
      val_list.append(new_val)
      val_mean_list.append(new_val.mean())
    xm.mark_step()
    xm.wait_device_ops()
    self.assertTrue("ExecuteTime" in met.metric_names() or
                    "EagerOpExecuteTime" in met.metric_names())


class TestDebuggingUtil(test_utils.XlaTestCase):

  @skipOnEagerDebug
  def test_get_xla_tensor_debug_info(self):
    device = xm.xla_device()
    # test non xla tensor
    cpu_t1 = torch.randn(5)
    cpu_t1_info = torch_xla._XLAC._get_xla_tensor_debug_info(cpu_t1)
    self.assertIn('Not a XLATensor', cpu_t1_info)

    # test a tensor with IR
    t1 = cpu_t1.to(device)
    t2 = t1 + 5
    t2_info = torch_xla._XLAC._get_xla_tensor_debug_info(t2)
    self.assertIn('XLA Shape: f32[5]', t2_info)
    self.assertIn('aten::add', t2_info)
    self.assertIn('XLAData: None', t2_info)

    # after makr_step XLAData should present
    xm.mark_step()
    t2_info_new = torch_xla._XLAC._get_xla_tensor_debug_info(t2)
    self.assertNotIn('XLAData: None', t2_info_new)
    self.assertIn('Data Shape: f32[5]', t2_info_new)
    self.assertIn('IR: None', t2_info_new)


class TestOpBuilder(test_utils.XlaTestCase):

  def runOpBuilderTest(self,
                       name,
                       tensors,
                       opfn,
                       aten_fn=None,
                       device=None,
                       rel_err=1e-2,
                       abs_err=1e-5,
                       kwargs=dict()):
    op = xor.register(name, opfn)
    if device is None:
      device = xm.xla_device()
    if aten_fn is None:
      aten_fn = opfn
    tensors = xu.as_list(tensors)
    xla_tensors = [
        x.to(device).detach().requires_grad_(x.requires_grad) for x in tensors
    ]
    results = xu.as_list(aten_fn(*tensors, **kwargs))
    xla_results = xu.as_list(op(*xla_tensors, **kwargs))
    self.compareResults(results, xla_results, rel_err=rel_err, abs_err=abs_err)

  def test_add(self):

    def op_fn(a, b, **kwargs):
      return a + b

    self.runOpBuilderTest(
        'test_add', [torch.randn(2, 2), torch.randn(2, 2)], op_fn)

  def test_mul(self):

    def op_fn(a, b, **kwargs):
      return a * b

    self.runOpBuilderTest(
        'test_mul', [torch.randn(2, 2), torch.randn(2, 2)], op_fn)

  def test_conditional(self):

    def op_fn(k, a, b, k0=None):

      def btrue(a, b):
        return a + b

      def bfalse(a, b):
        return a - b

      cond = k > xb.Op.scalar(k.builder(), k0, dtype=k.shape().dtype)
      return cond.mkconditional((a, b), btrue, bfalse)

    def aten_fn(k, a, b, k0=None):
      return a + b if k.item() > k0 else a - b

    self.runOpBuilderTest(
        'test_conditional',
        [torch.tensor(0.2),
         torch.randn(2, 2),
         torch.randn(2, 2)],
        op_fn,
        aten_fn=aten_fn,
        kwargs={'k0': 0.1})
    self.runOpBuilderTest(
        'test_conditional',
        [torch.tensor(0.2),
         torch.randn(2, 2),
         torch.randn(2, 2)],
        op_fn,
        aten_fn=aten_fn,
        kwargs={'k0': 0.9})

  def test_while(self):

    def op_fn(a, b, limit=None):

      def cond(counter, a, b):
        return counter < xb.Op.scalar(
            counter.builder(), limit, dtype=xb.Type.S32)

      def body(counter, a, b):
        next_counter = counter + xb.Op.scalar(
            counter.builder(), 1, dtype=xb.Type.S32)
        return xb.Op.tuple((next_counter, a + b, b))

      zero = xb.Op.scalar(a.builder(), 0, dtype=xb.Type.S32)
      w = xb.Op.mkwhile((zero, a, b), cond, body)
      return w.get_tuple_element(1)

    def aten_fn(a, b, limit=None):
      for _ in range(0, limit):
        a = a + b
      return a

    self.runOpBuilderTest(
        'test_while', [torch.randn(2, 2), torch.randn(2, 2)],
        op_fn,
        aten_fn=aten_fn,
        kwargs={'limit': 10})

  def test_triangular_solve(self):

    def op_fn(b, A, lower, unit_diagonal, transpose_a):
      return A.triangualr_solve(b, True, lower, unit_diagonal, transpose_a)

    def aten_fn(b, A, lower, unit_diagonal, transpose_a):
      return torch.triangular_solve(
          b,
          A,
          upper=not lower,
          unitriangular=unit_diagonal,
          transpose=transpose_a)

    self.runOpBuilderTest(
        'test_triangular_solve',
        [torch.randn(2, 3), torch.randn(2, 2).triu()],
        op_fn,
        aten_fn=aten_fn,
        kwargs={
            'lower': False,
            'unit_diagonal': False,
            'transpose_a': False
        })


class MpDecoratorTest(test_utils.XlaTestCase):

  @xtu.mp_test
  def test_mp_decorator(self):
    xla_device = xm.xla_device()
    self.assertTrue(xla_device.type == 'xla')


class XpTraceTest(test_utils.XlaTestCase):

  def test_non_empty_scope(self):
    with self.assertRaisesRegex(
        RuntimeError, r'Expecting scope to be empty but it is conv1.1'):
      with xp.Trace('conv1'):
        xm.mark_step()

  def test_non_empty_scope_decorator(self):

    @xp.trace_me("conv2")
    def func():
      xm.mark_step()

    with self.assertRaisesRegex(RuntimeError,
                                r'Expecting scope to be empty but it is conv2'):
      func()


class RegisterXLAKeyTest(test_utils.XlaTestCase):

  def test_multi_init_xla_backend(self):
    torch_xla._XLAC._init_xla_lazy_backend()
    torch_xla._XLAC._init_xla_lazy_backend()
    self.assertEqual(met.counter_value("RegisterXLAFunctions"), 1)


@unittest.skipIf(
    os.environ.get('XLA_USE_EAGER_DEBUG_MODE'),
    "Skipping test under XLA_USE_EAGER_DEBUG_MODE because `result` will not \
      reference a graph due to eager evaluation.")
class TestLoweringContext(test_utils.XlaTestCase):

  def test_api(self):
    device = xm.xla_device()
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)

    result = a + b

    ctx = torch_xla._XLAC.lowering.LoweringContext()
    ctx.build([result])
    hlo = ctx.hlo()
    hlo_text = ctx.hlo_text()
    self.assertTrue('opcode: "parameter"' in hlo_text)
    self.assertTrue('opcode: "add"' in hlo_text)
    mapping = ctx.parameter_id_tensor_mapping()
    self.assertEqual(len(mapping), 2)


class TestGeneric(test_utils.XlaTestCase):

  def test_zeros_like_patch(self):
    a = torch.ones(3, 3)
    b = torch.zeros_like(a, dtype=torch.int8)
    self.assertEqual(b.dtype, torch.int8)
    self.assertEqual(b.sum().item(), 0)

  def test_git_revisons(self):
    revs = torch_xla._XLAC._get_git_revs()
    self.assertTrue('xla' in revs)
    self.assertTrue(revs['xla'])
    self.assertTrue('torch' in revs)

  def test_send_to_device_grad(self):
    xla_device = xm.xla_device()
    t = _gen_tensor(2, 2, requires_grad=True)
    dt = xm.send_cpu_data_to_device([t], xla_device)
    self.assertTrue(dt[0].requires_grad)

  def test_send_to_device_single(self):
    xla_device = xm.xla_device()
    t = _gen_tensor(2, 2)
    dt = xm.send_cpu_data_to_device(t, xla_device)
    self.assertEqual(dt[0].device, xla_device)
    self.assertTrue(torch.all(torch.eq(dt[0].cpu(), t)))

  def test_util_foreach_api(self):

    class ForTest(object):

      def __init__(self):
        self.a = {'k': [1, 2, 3], 4.9: 'y', 5: {'a': 'n'}}
        self.b = ('f', 17)

    duped_data = ForTest()
    data = {
        2.3: 11,
        21: ForTest(),
        'w': [12, ForTest(), duped_data],
        123: duped_data,
    }

    ids = []

    def collect(x):
      ids.append(id(x))

    xu.for_each_instance(data, lambda x: isinstance(x, (int, str, float)),
                         collect)

    wids = []

    def convert(x):
      wids.append(id(x))
      return x

    xu.for_each_instance_rewrite(data,
                                 lambda x: isinstance(x, (int, str, float)),
                                 convert)
    self.assertEqual(len(ids), 17)
    self.assertEqual(ids, wids)

  def test_util_foreach_api_cycle(self):

    class ForTest1(object):

      def __init__(self, a, b):
        self.a = a
        self.b = b

    class ForTest2(object):

      def __init__(self, a):
        self.a = a
        self.b = ForTest1(self, a)

    xdata = {
        2: (11, ['a', 'b'], 17),
        'w': [12, 'q', 12.33],
        17.09: set(['a', 'b', 21]),
    }
    data = ForTest2(xdata)

    wids = []

    def convert(x):
      wids.append(id(x))
      return x

    xu.for_each_instance_rewrite(data,
                                 lambda x: isinstance(x, (int, str, float)),
                                 convert)
    self.assertEqual(len(wids), 11)

  def test_data_wrapper(self):

    class PackWrapper(xu.DataWrapper):

      def __init__(self, pack):
        super(PackWrapper, self).__init__()
        self.pack = pack

      def get_tensors(self):
        return [
            self.pack.data, self.pack.sorted_indices, self.pack.unsorted_indices
        ]

      def from_tensors(self, tensors):
        return nn.utils.rnn.PackedSequence(tensors[0], self.pack.batch_sizes,
                                           tensors[1], tensors[2])

    batch_in = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]],
                            dtype=torch.float32,
                            requires_grad=True).unsqueeze(-1)
    seq_lengths = [3, 2, 1]
    pack = torch.nn.utils.rnn.pack_padded_sequence(
        batch_in, seq_lengths, batch_first=True)

    wpack = PackWrapper(pack)

    xla_device = xm.xla_device()
    xdata = xm.send_cpu_data_to_device(wpack, xla_device)
    self.assertTrue(isinstance(xdata, nn.utils.rnn.PackedSequence))
    self.assertEqual(xdata.batch_sizes.device, torch.device('cpu'))
    self.assertEqual(xdata.data.device, xla_device)

  @skipIfFunctionalizationDisabled(
      "https://github.com/pytorch/xla/pull/7864#issuecomment-2294034008")
  def test_as_strided_input_larger(self):
    size = (5, 5)
    device = xm.xla_device()

    a = torch.ones(size, device=device)
    small_a = a[:, ::2]
    former_a = small_a.as_strided(size, (5, 1), 0)

    self.assertEqual(a, former_a)

  def _test_move_tensor_cuda_to_xla(self, cpu_tensor):
    # Assumes CPU-XLA data movement works.
    cuda_tensor = cpu_tensor.to("cuda")
    # Move tensor CUDA -> XLA.
    xla_tensor = cuda_tensor.to(xm.xla_device())
    # Move the XLA tensor back to CPU, and check that it is the same as
    # the original CPU tensor.
    self.assertTrue(torch.equal(cpu_tensor, xla_tensor.cpu()))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_aten_move_cuda_to_xla(self):
    self._test_move_tensor_cuda_to_xla(torch.arange(5))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_aten_move_scalar_cuda_to_xla(self):
    # 0-dimensional scalar-tensor
    # Has a different execution path than other tensors.
    self._test_move_tensor_cuda_to_xla(torch.tensor(42))

  def test_unsafe_buffer_pointer(self):
    xla_device = xm.xla_device()
    xla_tensor_0 = torch.tensor(42).to(xla_device)
    # `mark_step` ensures xtensor->CurrentDataHandle() != nullptr
    xm.mark_step()
    buf_ptr_0 = torch_xla._XLAC._unsafe_buffer_pointer(xla_tensor_0)
    self.assertGreaterEqual(buf_ptr_0, 0)

    # xtensor->CurrentDataHandle() == nullptr but xtensor->CurrentIrValue().node != nullptr and device_data != nullptr
    xla_tensor_1 = torch.tensor(42, device=xm.xla_device())
    buf_ptr_1 = torch_xla._XLAC._unsafe_buffer_pointer(xla_tensor_1)
    self.assertGreaterEqual(buf_ptr_1, 0)

    # xtensor->CurrentDataHandle() == nullptr but xtensor->CurrentIrValue().node != nullptr and device_data != nullptr
    xla_tensor_2 = torch.ones((5, 5)).to(xla_device)
    buf_ptr_2 = torch_xla._XLAC._unsafe_buffer_pointer(xla_tensor_2)
    self.assertGreaterEqual(buf_ptr_2, 0)

    xla_tensor_3 = torch.arange(5, device=xm.xla_device())
    xm.mark_step()
    # Without the `wait_device_ops()`, the pjrt buffer (pjrt_data->buffer) at https://github.com/pytorch/xla/blob/e3fc03314dab5f44e3ed9ccbba6c15fbca3285cd/torch_xla/csrc/runtime/pjrt_computation_client.cc#L467 will be nullptr.
    xm.wait_device_ops()
    buf_ptr_3 = torch_xla._XLAC._unsafe_buffer_pointer(xla_tensor_3)
    self.assertGreaterEqual(buf_ptr_3, 0)


class TestDLPack(parameterized.TestCase):

  def _test_dlpack_capsule_conversion_helper(self, xla_tensor):
    dlpt = xdlpack.to_dlpack(xla_tensor)  # dlpt1 has type PyCapsule
    xla_tensor2 = xdlpack.from_dlpack(dlpt)

    self.assertEqual(xla_tensor.device, xla_tensor2.device)
    self.assertTrue(torch.allclose(xla_tensor.cpu(), xla_tensor2.cpu()))
    self.assertRaisesRegex(RuntimeError,
                           "DLTensor capsule can be consumed only once",
                           lambda: xdlpack.from_dlpack(dlpt))

    self.assertEqual(
        torch_xla._XLAC._unsafe_buffer_pointer(xla_tensor),
        torch_xla._XLAC._unsafe_buffer_pointer(xla_tensor2))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  @parameterized.parameters(*all_types_and(torch.half, torch.bfloat16))
  def test_dlpack_roundtrip_tensor(self, dtype):
    xla_device = xm.xla_device()
    # xtensor->CurrentDataHandle() == nullptr but xtensor->CurrentIrValue().node != nullptr and device_data != nullptr
    # xla_tensor_2 uses XLANativeFunctions::_to_copy
    xla_tensor_2 = torch.arange(5, dtype=dtype).to(xla_device)
    self._test_dlpack_capsule_conversion_helper(xla_tensor_2)

    # xla_tensor_3 uses arange_out IR node.
    xla_tensor_3 = torch.arange(5, dtype=dtype, device=xm.xla_device())
    xm.mark_step()
    self._test_dlpack_capsule_conversion_helper(xla_tensor_3)

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  @parameterized.parameters(*all_types_and_complex_and(torch.half,
                                                       torch.bfloat16,
                                                       torch.bool, torch.uint16,
                                                       torch.uint32,
                                                       torch.uint64))
  def test_dlpack_roundtrip_scalar(self, dtype):
    xla_device = xm.xla_device()
    xla_tensor_0 = torch.tensor(42, dtype=dtype).to(xla_device)
    # `mark_step` ensures xtensor->CurrentDataHandle() != nullptr
    xm.mark_step()
    self._test_dlpack_capsule_conversion_helper(xla_tensor_0)

    xla_tensor_1 = torch.tensor(42, dtype=dtype).to(xla_device)
    # xtensor->CurrentDataHandle() == nullptr but xtensor->CurrentIrValue().node != nullptr and device_data != nullptr
    self._test_dlpack_capsule_conversion_helper(xla_tensor_1)

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_dlpack_roundtrip_bool(self):
    xla_tensor = torch.ones(1, dtype=torch.bool).to(xm.xla_device())
    self._test_dlpack_capsule_conversion_helper(xla_tensor)

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_dlpack_pytorch_cuda_to_xla(self):
    t1_cuda = torch.arange(5).cuda()
    dlt1 = torch.utils.dlpack.to_dlpack(t1_cuda)
    xla_t1 = xdlpack.from_dlpack(dlt1)
    self.assertEqual(xla_t1.device.type, 'xla')
    self.assertEqual(xla_t1.device.index, t1_cuda.device.index)
    t1_cuda[0] = t1_cuda[0] + 20
    self.assertTrue(torch.allclose(xla_t1.cpu(), t1_cuda.cpu()))

    t2_cuda = torch.tensor(5).cuda()
    dlt2 = torch.utils.dlpack.to_dlpack(t2_cuda)
    xla_t2 = xdlpack.from_dlpack(dlt2)
    self.assertEqual(xla_t2.device.type, 'xla')
    self.assertEqual(xla_t2.device.index, t2_cuda.device.index)
    t2_cuda.fill_(6)
    self.assertTrue(torch.allclose(xla_t2.cpu(), t2_cuda.cpu()))

    cuda1 = torch.device('cuda:1')
    t3_cuda = torch.tensor(5, device=cuda1)
    dlt3 = torch.utils.dlpack.to_dlpack(t3_cuda)
    xla_t3 = xdlpack.from_dlpack(dlt3)
    self.assertEqual(xla_t3.device.type, 'xla')
    self.assertEqual(
        xla_t3.device.index,
        t3_cuda.device.index,
        msg='both value should 1. xla_t3.device should be xla:1.')
    t3_cuda.fill_(6)
    self.assertTrue(torch.allclose(xla_t3.cpu(), t3_cuda.cpu()))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_dlpack_pytorch_cuda_to_xla_protocol_conversion(self):
    # Unlike the test_dlpack_pytorch_cuda_to_xla,
    # torch_cuda_tensor has attribute __dlpack__ and __dlpack_device__.
    # From cuda tensors to xla tensors, the synchronization is handdled implicitly.
    t1_cuda = torch.arange(5).cuda()
    xla_t1 = xdlpack.from_dlpack(t1_cuda)
    self.assertEqual(xla_t1.device.type, 'xla')
    self.assertEqual(xla_t1.device.index, t1_cuda.device.index)
    t1_cuda[0] = t1_cuda[0] + 20
    self.assertTrue(torch.allclose(xla_t1.cpu(), t1_cuda.cpu()))

    t2_cuda = torch.tensor(5).cuda()
    xla_t2 = xdlpack.from_dlpack(t2_cuda)
    self.assertEqual(xla_t2.device.type, 'xla')
    self.assertEqual(xla_t2.device.index, t2_cuda.device.index)
    t2_cuda.fill_(6)
    self.assertTrue(torch.allclose(xla_t2.cpu(), t2_cuda.cpu()))

    cuda1 = torch.device('cuda:1')
    t3_cuda = torch.tensor(5, device=cuda1)
    xla_t3 = xdlpack.from_dlpack(t3_cuda)
    self.assertEqual(xla_t3.device.type, 'xla')
    self.assertEqual(
        xla_t3.device.index,
        t3_cuda.device.index,
        msg='both value should 1. xla_t3.device should be xla:1.')
    t3_cuda.fill_(6)
    self.assertTrue(torch.allclose(xla_t3.cpu(), t3_cuda.cpu()))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_dlpack_xla_to_pytorch_cuda(self):
    xla_t1 = torch.arange(5).to(xm.xla_device())
    dlt1 = xdlpack.to_dlpack(xla_t1)
    cuda_t1 = torch.utils.dlpack.from_dlpack(dlt1)
    self.assertEqual(cuda_t1.device.type, 'cuda')
    self.assertEqual(cuda_t1.device.index, xla_t1.device.index)
    cuda_t1[0] = cuda_t1[0] + 20
    self.assertTrue(torch.allclose(xla_t1.cpu(), cuda_t1.cpu()))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_dlpack_xla_to_pytorch_cuda_protocol_conversion(self):
    xla_t1 = torch.arange(5).to(xm.xla_device())
    caps_t1 = torch.utils.dlpack.to_dlpack(xla_t1)
    cuda_t1 = torch.utils.dlpack.from_dlpack(caps_t1)
    self.assertEqual(cuda_t1.device.type, 'cuda')
    self.assertEqual(cuda_t1.device.index, xla_t1.device.index)
    cuda_t1[0] = cuda_t1[0] + 20
    self.assertTrue(torch.allclose(xla_t1.cpu(), cuda_t1.cpu()))

  @onlyIfTorchSupportsCUDA
  @onlyIfPJRTDeviceIsCUDA
  def test_dlpack_non_default_layout(self):
    cuda_t = torch.arange(25, device=torch.device('cuda')).reshape(5, 5)

    t1 = cuda_t.t()
    xla_t1 = xdlpack.from_dlpack(t1.__dlpack__())
    self.assertEqual(xla_t1.device.type, 'xla')
    self.assertEqual(xla_t1.device.index, t1.device.index)
    self.assertTrue(torch.allclose(t1.cpu(), xla_t1.cpu()))

    t2 = cuda_t[0]
    xla_t2 = xdlpack.from_dlpack(t2.__dlpack__())
    self.assertEqual(xla_t2.device.type, 'xla')
    self.assertEqual(xla_t2.device.index, t2.device.index)
    self.assertTrue(torch.allclose(t2.cpu(), xla_t2.cpu()))

    t3 = cuda_t[:, 0]
    self.assertRaisesRegex(
        RuntimeError,
        r"Only DLPack tensors with trivial \(compact\) striding are supported",
        lambda: xdlpack.from_dlpack(t3.__dlpack__()))

    t4 = cuda_t[1, :]
    xla_t4 = xdlpack.from_dlpack(t4.__dlpack__())
    self.assertEqual(xla_t4.device.type, 'xla')
    self.assertEqual(xla_t4.device.index, t4.device.index)
    self.assertTrue(torch.allclose(t4.cpu(), xla_t4.cpu()))

    t5 = cuda_t[1]
    xla_t5 = xdlpack.from_dlpack(t5.__dlpack__())
    self.assertEqual(xla_t5.device.type, 'xla')
    self.assertEqual(xla_t5.device.index, t5.device.index)
    self.assertTrue(torch.allclose(t5.cpu(), xla_t5.cpu()))


class SimpleModelWithDropout(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.x = torch.nn.Linear(128, 128)
    self.register_buffer("buffer", torch.zeros(64, 64))
    self.dropout = torch.nn.Dropout(p=0.1)
    self.to_save = []

  def save_output(self, output):
    self.to_save.append(output.detach().cpu())

  def forward(self, inp):
    x = self.x(inp)
    output = self.dropout(x)
    xm.add_step_closure(self.save_output, args=(output,), run_async=False)
    return output


class TestActivationCheckpoint(test_utils.XlaTestCase):

  def test_dropout(self):
    device = xm.xla_device()
    model = SimpleModelWithDropout().to(device)
    model = checkpoint_module(model)
    _input = torch.randn(128, 128, requires_grad=True)
    _input = _input.to(device)
    output = model(_input)
    output = torch.sum(output)
    output.backward()
    xm.mark_step()
    same_output = torch.allclose(model.to_save[0], model.to_save[1])
    self.assertTrue(same_output,
                    f"in fwd {model.to_save[0]}, in bwd {model.to_save[1]}")

  def test_opt_barrier(self):
    device = xm.xla_device()
    model = SimpleModelWithDropout().to(device)
    model = checkpoint_module(model)
    _input = torch.randn(128, 128, requires_grad=True)
    _input = _input.to(device)
    output = model(_input)
    output = torch.sum(output)
    output.backward()

    hlo = torch_xla._XLAC._get_xla_tensors_hlo([model.x.weight.grad])
    lines = hlo.splitlines()
    opt_barrier = ""
    for line in lines:
      if "opt-barrier" in line:
        opt_barrier = line
        break

    # Somehow the CPU/GPU CI will not have the opt-barrier.
    if opt_barrier != "":
      self.assertEqual(opt_barrier.count("f32[128,128]"), 6)
      self.assertEqual(opt_barrier.count("f32[128]"), 2)
      self.assertEqual(opt_barrier.count("f32[64,64]"), 2)


# These tests were extracted and adapted from torchvision.
# Source: vision/test/test_ops.py
@onlyIfXLAExperimentalContains("nms")
class TestNMS(test_utils.XlaTestCase):

  def _reference_nms(self, boxes, scores, iou_threshold):
    import torchvision
    return torchvision.ops.nms(boxes.cpu(), scores.cpu(), iou_threshold)

  def _nms(self, boxes, scores, iou_threshold):
    import torchvision
    device = xm.xla_device()
    return torchvision.ops.nms(
        boxes.to(device), scores.to(device), iou_threshold).cpu()

  def _create_tensors_with_iou(self, N, iou_thresh):
    # force last box to have a pre-defined iou with the first box
    # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
    # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
    # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
    # Adjust the threshold upward a bit with the intent of creating
    # at least one box that exceeds (barely) the threshold and so
    # should be suppressed.
    boxes = torch.rand(N, 4) * 100
    boxes[:, 2:] += boxes[:, :2]
    boxes[-1, :] = boxes[0, :]
    x0, y0, x1, y1 = boxes[-1].tolist()
    iou_thresh += 1e-5
    boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
    scores = torch.rand(N)
    return boxes, scores

  @skipOnEagerDebug
  def test_nms_ref(self):

    def _test(iou, seed):
      torch.random.manual_seed(seed)
      err_msg = "NMS incompatible between CPU and reference implementation for IoU={}"
      boxes, scores = self._create_tensors_with_iou(1000, iou)
      keep_ref = self._reference_nms(boxes, scores, iou)
      keep = self._nms(boxes, scores, iou)
      self.assertEqual(keep, keep_ref, message=err_msg.format(iou))

    for iou in (0.2, 0.5, 0.8):
      for seed in range(10):
        with self.subTest(iou=iou, seed=seed):
          _test(iou, seed)

  def test_nms_input_errors(self):
    with self.assertRaisesRegex(RuntimeError, "boxes should be a 2D tensor."):
      self._nms(torch.rand(4), torch.rand(3), 0.5)
    with self.assertRaisesRegex(
        RuntimeError, "boxes should be a 2D tensor of shape \[N, 4\]."):
      self._nms(torch.rand(3, 5), torch.rand(3), 0.5)
    with self.assertRaisesRegex(RuntimeError, "scores should be a 1D tensor."):
      self._nms(torch.rand(3, 4), torch.rand(3, 2), 0.5)
    with self.assertRaisesRegex(
        RuntimeError,
        "boxes and scores should have the same size for dimension 0."):
      self._nms(torch.rand(3, 4), torch.rand(4), 0.5)

  def test_legacy(self):
    BOXES = (
        (0, 0, 3, 2),
        (3, 3, 11, 7),
        (2, 2, 5, 7),
        (7, 4, 15, 12),
    )
    SCORES = (0.9, 0.5, 0.95, 0.4)
    IOU_THRESHOLD = 0.08

    def fn(boxes, scores):
      return self._reference_nms(boxes, scores, IOU_THRESHOLD)

    boxes = torch.tensor(BOXES, dtype=torch.float)
    scores = torch.tensor(SCORES, dtype=torch.float)
    self.runAtenTest((boxes, scores), fn)


class TestHelperFunction(test_utils.XlaTestCase):

  def test_repeat_truncated(self):
    from torch_xla.experimental.custom_kernel import repeat_with_fixed_output_size
    met.clear_all()
    device = torch_xla.device()
    total_repeat_length = 20
    input = torch.randn(10).to(device)
    repeats = torch.tensor([0, 1, 2, 0, 4, 0, 6, 7, 8, 9]).to(device)
    res = repeat_with_fixed_output_size(input, repeats, total_repeat_length)
    # make sure there is no graph break
    assert 'aten::' not in met.short_metrics_report()
    expected = torch.repeat_interleave(input, repeats)[:total_repeat_length]
    self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))

  def test_repeat_extended(self):
    from torch_xla.experimental.custom_kernel import repeat_with_fixed_output_size
    met.clear_all()
    device = torch_xla.device()
    total_repeat_length = 100
    input = torch.randn(10).to(device)
    repeats = torch.tensor([0, 5, 2, 0, 4, 9, 6, 7, 8, 0]).to(device)
    res = repeat_with_fixed_output_size(input, repeats, total_repeat_length)
    # make sure there is no graph break
    assert 'aten::' not in met.short_metrics_report()
    base = torch.repeat_interleave(input, repeats)[:total_repeat_length]
    # remaining space will be filled with last value in `input`.
    expected = torch.cat(
        (base,
         torch.repeat_interleave(input[-1],
                                 total_repeat_length - base.size()[0])))
    self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))

  def test_repeat_special(self):
    from torch_xla.experimental.custom_kernel import repeat_with_fixed_output_size
    met.clear_all()
    device = torch_xla.device()
    total_repeat_length = 135
    num_groups = 8
    input = torch.arange(num_groups, dtype=torch.int32).to(device)
    repeats = torch.tensor([3, 6, 2, 14, 27, 47, 8, 28]).to(device)
    res = repeat_with_fixed_output_size(input, repeats, total_repeat_length)
    # make sure there is no graph break
    assert 'aten::' not in met.short_metrics_report()
    expected = torch.repeat_interleave(input, repeats)[:total_repeat_length]
    self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))


if __name__ == '__main__':
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(met.metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
