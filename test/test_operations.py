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
# Setup import folders.
_XLA_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.join(os.path.dirname(_XLA_FOLDER), 'test'))
sys.path.insert(0, _XLA_FOLDER)

# Normal imports section starts here.
import collections
from common_utils import TestCase, run_tests, iter_indices
import copy
import itertools
import numpy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.model_comparator as mc
import torch_xla_py.parallel_loader as pl
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torchvision
import unittest

DeviceSupport = collections.namedtuple('DeviceSupport', ['num_devices'])


def _gen_tensor(*args, **kwargs):
  return torch.randn(*args, **kwargs)


def _gen_int_tensor(*args, **kwargs):
  return torch.randint(*args, **kwargs)


class Holder(object):
  pass


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


def _dump_differences(target, result, rtol=1e-5, atol=1e-3, max_diff_count=0):
  env = Holder()
  env.max_diff = 0.0
  env.max_rel = None
  env.max_index = None
  env.diff_count = 0

  def check_values(a, b, index):
    r = max(abs(a), abs(b)) * rtol
    diff = abs(a - b)
    if diff > max(r, atol):
      print('a={}\tb={}\tdiff={}\tindex={}'.format(a, b, diff, index))
      env.diff_count += 1
      if diff > env.max_diff:
        env.max_diff = diff
        env.max_rel = diff / max(abs(a), abs(b))
        env.max_index = index

  if isinstance(target, torch.Tensor):
    assert isinstance(result, torch.Tensor)
    assert target.size() == result.size()
    if target.dim() > 0:
      for i in iter_indices(target):
        check_values(target[i], result[i], i)
        if max_diff_count > 0 and env.diff_count >= max_diff_count:
          break
    else:
      check_values(target.item(), result.item(), 0)
  elif isinstance(target, (list, tuple)):
    assert isinstance(result, (list, tuple))
    assert len(target) == len(result)
    for i, v in enumerate(target):
      check_values(v, result[i], [i])
      if max_diff_count > 0 and env.diff_count >= max_diff_count:
        break
  elif isinstance(target, float):
    assert isinstance(result, float)
    check_values(target, result, [])
  if env.max_index is not None:
    print('\nmax_diff={}\tmax_rel={}\tindex={}'.format(env.max_diff,
                                                       env.max_rel,
                                                       env.max_index))


class XlaTestCase(TestCase):

  def assertEqualRel(self, out, expected, rel_err=1e-2, abs_err=1e-5):
    try:
      diff_tensor = (out - expected).abs().float()
      max_rel_err = torch.max(out.abs(), expected.abs()).float() * rel_err
      # Allow higher relative differences as long as we're still below the
      # absolute error.
      max_abs_err = torch.max(max_rel_err,
                              torch.ones_like(out).float() * abs_err)
      super(XlaTestCase, self).assertEqual(diff_tensor.size(),
                                           max_abs_err.size())
      if (diff_tensor.numel() > 0 and
          torch.le(diff_tensor, max_abs_err).min().item() == 0):
        self.fail('Relative error higher than the maximum tolerance')
    except:
      _dump_differences(
          out,
          expected,
          rtol=rel_err,
          atol=abs_err,
          max_diff_count=FLAGS.max_diff_count)
      raise

  def assertEqualDbg(self, out, expected):
    try:
      super(XlaTestCase, self).assertEqual(out, expected)
    except:
      _dump_differences(
          out,
          expected,
          rtol=1e-8,
          atol=1e-8,
          max_diff_count=FLAGS.max_diff_count)
      raise

  def makeComparable(self, value):
    if isinstance(value, torch.Tensor):
      if value.dtype == torch.bool:
        value = value.to(dtype=torch.uint8)
      if xm.is_xla_tensor(value.data):
        return value.data.cpu()
      return value.data
    return value

  def runAtenTest(self, tensors, fn, device=None, rel_err=1e-2, abs_err=1e-5):
    if device is None:
      device = xm.xla_device()
    tensors = xu.as_list(tensors)
    xla_tensors = [x.to(device) for x in tensors]
    results = xu.as_list(fn(*tensors))
    xla_results = xu.as_list(fn(*xla_tensors))
    for at, xt in zip(results, xla_results):
      self.assertEqualRel(
          self.makeComparable(xt),
          self.makeComparable(at),
          rel_err=rel_err,
          abs_err=abs_err)


class TestToXlaTensorArena(XlaTestCase):

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


class TestParallelLoader(XlaTestCase):

  def test(self):
    devices = xm.get_xla_supported_devices()
    A = 3.11
    B = 4.09
    batch_size = 128 * len(devices)
    gen = xu.FnDataGenerator(
        lambda x: x * A + B, batch_size, _gen_tensor, dims=[8], count=10)
    para_loader = pl.ParallelLoader(gen, batch_size, devices)
    for x, (data, target) in para_loader:
      for device in devices:
        dx = para_loader.to(data, device)
        self.assertEqual(dx.device, torch.device(device))


class TestAtenTensorTo(XlaTestCase):

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


class TestParallelTensorMNIST(XlaTestCase):

  def test(self):
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

      for x, (data, target) in loader:
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


class TestParallelTensorResnet18(XlaTestCase):

  def test(self):
    devices = xm.get_xla_supported_devices()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=4)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(batch_size, 3, 224,
                          224), torch.zeros(batch_size, dtype=torch.int64)),
        sample_count=sample_count * len(devices))

    def loop_fn(model, loader, device, context):
      loss_fn = nn.NLLLoss()
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

      for x, (data, target) in loader:
        with xu.TimedScope(msg='Training loop: ', printfn=None):
          optimizer.zero_grad()
          output = xu.timed(lambda: model(data), msg='Model: ', printfn=None)
          loss = xu.timed(
              lambda: loss_fn(output, target), msg='Loss: ', printfn=None)
          xu.timed(loss.backward, msg='LossBkw: ', printfn=None)
          xu.timed(
              lambda: xm.optimizer_step(optimizer), msg='Step: ', printfn=None)
          self.assertLess(loss.cpu().item(), 3.0)

    model_parallel = dp.DataParallel(
        torchvision.models.resnet18, device_ids=devices)
    model_parallel(loop_fn, train_loader)


class TestLongGraphChain(XlaTestCase):

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
    self.assertEqualRel(x, xla_x.cpu(), rel_err=1e-3, abs_err=5)


class TestSelect(XlaTestCase):

  def test_get_xla_tensor(self):
    x = _gen_tensor(14, 24, 8, device=xm.xla_device())
    t = x.data.cpu()
    sx = x.select(1, 12)
    tx = t.select(1, 12)
    self.assertEqual(tx, sx.data.cpu())


class TestAtenXlaTensor(XlaTestCase):

  def test_get_real_xla_devices(self):
    devices = xm.get_xla_supported_devices()
    xla_devices = torch_xla._XLAC._xla_real_devices(devices)
    for device, xdevice in zip(devices, xla_devices):
      self.assertTrue(re.match(r'(CPU|GPU|TPU):\d+$', xdevice) is not None)

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

  def test_masked_fill_with_tensor(self):
    input = _gen_tensor(2, 5, 4, 3)
    mask = torch.randint(0, 2, input.size(), dtype=torch.bool)
    value = torch.tensor(42)
    xla_input = input.to(xm.xla_device())
    xla_mask = mask.to(xm.xla_device())
    xla_value = value.to(xm.xla_device())
    result = torch.masked_fill(input, mask, value)
    xla_result = torch.masked_fill(xla_input, xla_mask, xla_value)
    self.assertEqual(input.data, xla_input.data.cpu())
    self.assertEqual(result.data, xla_result.data.cpu())

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
    x = torch.randperm(3, device=xm.xla_device())
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
            torch.zeros(shape, device=xm.xla_device(), dtype=torch.uint8),
            6, 10)
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

  def test_empty_advanced_indexing(self):
    xla_device = xm.xla_device()
    base = torch.randn(2, 3, 4, 5)
    xla_base = base.to(device=xla_device)
    result = base[:, torch.empty(0, 6, dtype=torch.int64)]
    xla_result = xla_base[:, torch.empty(0, 6, dtype=torch.int64)]
    self.assertEqual(result, xla_result)

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
    self.assertEqual(grad_grad_input, xla_grad_grad_input)

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

  def test_index_put(self):
    xla_device = xm.xla_device()
    a = torch.tensor([1, 1, 1, 1]).to(xla_device).to(dtype=torch.float32)
    b = torch.rand(4) > 0.1
    a[b] = 10
    vset = b.sum().item()
    self.assertEqual(a.sum().item(), 10.0 * vset + (4.0 - vset))

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

    # PRED cannot be automatically promoted to other dtypes.
    # Test both cpu and xla here to make sure we get notified
    # once PyTorch changes its behavior.
    self.assertRaises(RuntimeError, lambda: c & c.byte())
    self.assertRaises(RuntimeError, lambda: c + c.byte())
    self.assertRaises(RuntimeError, lambda: xla_c & xla_c.byte())
    # RuntimeError is trigger in execution instead of lowering,
    # thus print is required.
    self.assertRaises(RuntimeError, lambda: print(xla_c + xla_c.byte()))

  def test_bitwise_type(self):
    xla_device = xm.xla_device()
    a = torch.randint(255, (4,), dtype=torch.long)
    xla_a = a.to(xla_device)
    self.assertRaises(RuntimeError, lambda: a & a.byte())
    self.assertRaises(RuntimeError, lambda: xla_a & xla_a.byte())

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

  def test_norm_p0(self):
    # p = 0 is equivalent to nonzero
    xla_device = xm.xla_device()
    a = torch.randn(3, 2)
    xla_a = a.to(xla_device)
    norm = a.norm(p = 0)
    xla_norm = xla_a.norm(p = 0)
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

  def test_pred_and_u8(self):

    def test_fn(a):
      return torch.isfinite(a) & a.ne(0)

    self.runAtenTest(torch.rand(4, 3), test_fn)

  def test_scatter_add_bool(self):
    xla_device = xm.xla_device()
    a = torch.tensor([[True, True, True, True, True], [True, True, True, True, True]])
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

  def test_max_throw(self):
    xla_device = xm.xla_device()
    xla_a = torch.randn(2, 0, 4, device=xla_device)
    self.assertRaises(RuntimeError, lambda: torch.max(xla_a, dim=1))
    self.assertRaises(RuntimeError, lambda: torch.max(xla_a))

  def test_save(self):
    xla_device = xm.xla_device()
    x = torch.randn(5, device=xla_device)
    x_file = tempfile.mktemp()
    try:
      torch.save(x, x_file)
      x_loaded = torch.load(x_file)
      self.assertEqual(x, x_loaded)
    finally:
      os.remove(x_file)

  def test_print(self):
    xla_device = xm.xla_device()
    x = torch.tensor([5], device=xla_device)
    expected_str = 'tensor([5], device=\'' + str(xla_device) + '\')'
    self.assertExpectedInline(str(x), expected_str)


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


class TestModelComparator(XlaTestCase):

  def test(self):
    xla_device = xm.xla_device()
    x = _gen_tensor(8, 1, 28, 28)

    torch.manual_seed(42)
    model = MNISTComparator()
    save_dir1 = xu.TmpFolder()
    mc.configure(save_dir1.name)
    model(x)

    save_dir2 = xu.TmpFolder()
    mc.configure(save_dir2.name)
    torch.manual_seed(42)
    xla_model = MNISTComparator().to(xla_device)
    xla_x = x.to(xla_device)
    xla_model(xla_x)

    report = mc.compare(save_dir1.name, save_dir2.name, rtol=1e-03, atol=1e-04)
    if report:
      print(report)
    self.assertEqual(len(report), 0)


class TestGeneric(XlaTestCase):

  def test_zeros_like_patch(self):
    a = torch.ones(3, 3)
    b = torch.zeros_like(a, dtype=torch.int8)
    self.assertEqual(b.dtype, torch.int8)
    self.assertEqual(b.sum().item(), 0)


if __name__ == '__main__':
  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(torch_xla._XLAC._xla_metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
