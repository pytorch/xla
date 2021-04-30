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
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_op_registry as xor
import torch_xla.distributed.data_parallel as dp
import torch_xla.debug.metrics as met
import torch_xla.debug.model_comparator as mc
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as xtu
import torch_xla.utils.utils as xu
import torch_xla.utils.serialization as xser
import torch_xla.core.xla_model as xm
import torch_xla.core.functions as xf
import torchvision
import unittest

DeviceSupport = collections.namedtuple('DeviceSupport', ['num_devices'])


def _gen_tensor(*args, **kwargs):
  return torch.randn(*args, **kwargs)


def _gen_int_tensor(*args, **kwargs):
  return torch.randint(*args, **kwargs)


def _gen_mask(size):
  return torch.randint(0, 2, size, dtype=torch.bool)


class Holder(object):
  pass


def _iter_indices(tensor):
  if tensor.dim() == 0:
    return range(0)
  if tensor.dim() == 1:
    return range(tensor.size(0))
  return itertools.product(*(range(s) for s in tensor.size()))


def _is_iterable(obj):
  try:
    iter(obj)
    return True
  except TypeError:
    return False


def _set_rng_seed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  numpy.random.seed(seed)
  xm.set_rng_state(seed)


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


def _prepare_tensors_for_diff(ta, tb):
  a = ta.to(device='cpu')
  b = tb.to(device='cpu')
  if a.dtype == torch.float16 or a.dtype == torch.bfloat16:
    a = a.to(torch.float32)
  if b.dtype == torch.float16 or b.dtype == torch.bfloat16:
    b = b.to(torch.float32)
  if b.dtype != a.dtype:
    b = b.to(a.dtype)
  if xu.getenv_as('TEST_PRINT_TENSORS', bool, defval=False):
    print('Tensor A ({}):\n{}'.format(ta.device, a), file=sys.stderr)
    print('Tensor B ({}):\n{}'.format(tb.device, b), file=sys.stderr)
  return a, b


def _dump_differences(target, result, rtol=1e-5, atol=1e-3, max_diff_count=0):
  env = Holder()
  env.max_diff = 0.0
  env.max_rel = None
  env.max_index = None
  env.diff_count = 0

  def check_values(a, b, index):
    a, b = _prepare_tensors_for_diff(a, b)
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
      for i in _iter_indices(target):
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


class XlaTestCase(unittest.TestCase):
  PRECISION = 1e-5
  STRING_CLASSES = (str, bytes)

  def __init__(self, method_name='runTest'):
    super(XlaTestCase, self).__init__(method_name)

  def setUp(self):
    _set_rng_seed(1234)

  def safeCoalesce(self, t):
    tc = t.coalesce()
    self.assertEqual(tc.to_dense(), t.to_dense())
    self.assertTrue(tc.is_coalesced())
    # Our code below doesn't work when nnz is 0, because
    # then it's a 0D tensor, not a 2D tensor.
    if t._nnz() == 0:
      self.assertEqual(t._indices(), tc._indices())
      self.assertEqual(t._values(), tc._values())
      return tc

    value_map = {}
    for idx, val in zip(t._indices().t(), t._values()):
      idx_tup = tuple(idx.tolist())
      if idx_tup in value_map:
        value_map[idx_tup] += val
      else:
        value_map[idx_tup] = val.clone() if isinstance(val,
                                                       torch.Tensor) else val

    new_indices = sorted(list(value_map.keys()))
    new_values = [value_map[idx] for idx in new_indices]
    if t._values().ndimension() < 2:
      new_values = t._values().new(new_values)
    else:
      new_values = torch.stack(new_values)
    new_indices = t._indices().new(new_indices).t()
    tg = t.new(new_indices, new_values, t.size())

    self.assertEqual(tc._indices(), tg._indices())
    self.assertEqual(tc._values(), tg._values())
    if t.is_coalesced():
      self.assertEqual(tc._indices(), t._indices())
      self.assertEqual(tc._values(), t._values())

    return tg

  # This has been copied from pytorch/test/common_utils.py in order to decouple
  # PyTorch/XLA tests from pytorch tests. We use this API only with a very
  # limited set of object types, so it could be eventually simplified.
  def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
    if isinstance(prec, str) and message == '':
      message = prec
      prec = None
    if prec is None:
      prec = self.PRECISION
    if isinstance(x, torch.Tensor) and isinstance(y, Number):
      self.assertEqual(
          x.item(), y, prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(y, torch.Tensor) and isinstance(x, Number):
      self.assertEqual(
          x, y.item(), prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(x, torch.Tensor) and isinstance(y, numpy.bool_):
      self.assertEqual(
          x.item(), y, prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(y, torch.Tensor) and isinstance(x, numpy.bool_):
      self.assertEqual(
          x, y.item(), prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

      def assertTensorsEqual(a, b):
        super(XlaTestCase, self).assertEqual(a.size(), b.size(), message)
        if a.numel() > 0:
          a, b = _prepare_tensors_for_diff(a, b)
          if (a.dtype == torch.bool) != (b.dtype == torch.bool):
            raise TypeError('Was expecting both tensors to be bool type.')
          else:
            if a.dtype == torch.bool and b.dtype == torch.bool:
              # we want to respect precision but as bool doesn't support substraction,
              # boolean tensor has to be converted to int
              a = a.to(torch.int)
              b = b.to(torch.int)

            diff = a - b
            if a.is_floating_point():
              # check that NaNs are in the same locations
              nan_mask = torch.isnan(a)
              self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
              diff[nan_mask] = 0
              # inf check if allow_inf=True
              if allow_inf:
                inf_mask = torch.isinf(a)
                inf_sign = inf_mask.sign()
                self.assertTrue(
                    torch.equal(inf_sign,
                                torch.isinf(b).sign()), message)
                diff[inf_mask] = 0
            # TODO: implement abs on CharTensor (int8)
            if diff.is_signed() and diff.dtype != torch.int8:
              diff = diff.abs()
            max_err = diff.max()
            self.assertLessEqual(max_err, prec, message)

      super(XlaTestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
      super(XlaTestCase, self).assertEqual(x.is_quantized, y.is_quantized,
                                           message)
      if x.is_sparse:
        x = self.safeCoalesce(x)
        y = self.safeCoalesce(y)
        assertTensorsEqual(x._indices(), y._indices())
        assertTensorsEqual(x._values(), y._values())
      elif x.is_quantized and y.is_quantized:
        self.assertEqual(
            x.qscheme(),
            y.qscheme(),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
        if x.qscheme() == torch.per_tensor_affine:
          self.assertEqual(
              x.q_scale(),
              y.q_scale(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
          self.assertEqual(
              x.q_zero_point(),
              y.q_zero_point(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
        elif x.qscheme() == torch.per_channel_affine:
          self.assertEqual(
              x.q_per_channel_scales(),
              y.q_per_channel_scales(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
          self.assertEqual(
              x.q_per_channel_zero_points(),
              y.q_per_channel_zero_points(),
              prec=prec,
              message=message,
              allow_inf=allow_inf)
          self.assertEqual(
              x.q_per_channel_axis(),
              y.q_per_channel_axis(),
              prec=prec,
              message=message)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(
            x.int_repr().to(torch.int32),
            y.int_repr().to(torch.int32),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
      else:
        assertTensorsEqual(x, y)
    elif isinstance(x, self.STRING_CLASSES) and isinstance(
        y, self.STRING_CLASSES):
      super(XlaTestCase, self).assertEqual(x, y, message)
    elif type(x) == set and type(y) == set:
      super(XlaTestCase, self).assertEqual(x, y, message)
    elif isinstance(x, dict) and isinstance(y, dict):
      if isinstance(x, collections.OrderedDict) and isinstance(
          y, collections.OrderedDict):
        self.assertEqual(
            x.items(),
            y.items(),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
      else:
        self.assertEqual(
            set(x.keys()),
            set(y.keys()),
            prec=prec,
            message=message,
            allow_inf=allow_inf)
        key_list = list(x.keys())
        self.assertEqual([x[k] for k in key_list], [y[k] for k in key_list],
                         prec=prec,
                         message=message,
                         allow_inf=allow_inf)
    elif _is_iterable(x) and _is_iterable(y):
      super(XlaTestCase, self).assertEqual(len(x), len(y), message)
      for x_, y_ in zip(x, y):
        self.assertEqual(
            x_, y_, prec=prec, message=message, allow_inf=allow_inf)
    elif isinstance(x, bool) and isinstance(y, bool):
      super(XlaTestCase, self).assertEqual(x, y, message)
    elif isinstance(x, Number) and isinstance(y, Number):
      if abs(x) == math.inf or abs(y) == math.inf:
        if allow_inf:
          super(XlaTestCase, self).assertEqual(x, y, message)
        else:
          self.fail('Expected finite numeric values - x={}, y={}'.format(x, y))
        return
      super(XlaTestCase, self).assertLessEqual(abs(x - y), prec, message)
    else:
      super(XlaTestCase, self).assertEqual(x, y, message)

  def assertEqualRel(self, out, expected, rel_err=1e-2, abs_err=1e-5):
    try:
      out, expected = _prepare_tensors_for_diff(out, expected)
      nan_mask = torch.isnan(expected)
      self.assertTrue(torch.equal(nan_mask, torch.isnan(out)))
      out[nan_mask] = 0
      expected[nan_mask] = 0
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
          expected,
          out,
          rtol=rel_err,
          atol=abs_err,
          max_diff_count=FLAGS.max_diff_count)
      raise

  def assertEqualDbg(self, out, expected):
    try:
      super(XlaTestCase, self).assertEqual(out, expected)
    except:
      _dump_differences(
          expected,
          out,
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

  def maybePrintGraph(self, tensors):
    env = os.environ.get('TEST_PRINT_GRAPH', '').lower()
    if env:
      if env == 'text':
        print(
            'Test Graph:\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_text(tensors)),
            file=sys.stderr)
      elif env == 'hlo':
        print(
            'Test Graph:\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_hlo(tensors)),
            file=sys.stderr)
      else:
        raise RuntimeError('Invalid TEST_PRINT_GRAPH value: {}'.format(env))

  def compareResults(self, results, xla_results, rel_err=1e-2, abs_err=1e-5):
    self.maybePrintGraph(xla_results)
    for at, xt in zip(results, xla_results):
      self.assertEqualRel(
          self.makeComparable(xt),
          self.makeComparable(at),
          rel_err=rel_err,
          abs_err=abs_err)

  def runAtenTest(self, tensors, fn, device=None, rel_err=1e-2, abs_err=1e-5):
    if device is None:
      device = xm.xla_device()
    tensors = xu.as_list(tensors)
    xla_tensors = [
        x.to(device).detach().requires_grad_(x.requires_grad) for x in tensors
    ]
    results = xu.as_list(fn(*tensors))
    xla_results = xu.as_list(fn(*xla_tensors))
    self.compareResults(results, xla_results, rel_err=rel_err, abs_err=abs_err)


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


class TestRandom(XlaTestCase):

  def test_random_from_to_bool(self):
    for from_val, to_val in [[0, 1], [0, 2], [1, 2]]:
      x = _gen_tensor(10, device=xm.xla_device())
      x.random_(from_val, to_val)
      delta = 1
      self.assertTrue(from_val <= x.to(torch.int).min() < (from_val + delta))
      self.assertTrue((to_val - delta) <= x.to(torch.int).max() < to_val)


class TestBinaryCrossEntropyLimitValue(XlaTestCase):

  def test_cross_entropy_loss(self):

    def test_fn(pred, target):
      lossfn = nn.BCELoss()
      return lossfn(pred, target)

    pred = torch.tensor(1.0)
    target = torch.tensor(1.0)
    for offset in [1, 0, 1e-8, 1e-7]:
      self.runAtenTest([pred - offset, target], test_fn)


class TestNllLossLimitValue(XlaTestCase):

  def test_nll_loss(self):

    def test_fn(logits, target):
      return nn.functional.nll_loss(logits, target)

    inf = float('inf')
    logits = torch.tensor([[1., inf], [-inf, 2.]])
    target = torch.tensor([0, 1])
    self.runAtenTest([logits, target], test_fn)


class TestInterOpSyncTensors(XlaTestCase):

  def test_inter_op_sync(self):

    def test_fn(x):
      # logaddexp can be replaced with any op that does not have
      # xla lowering.
      y = torch.logaddexp(x, x)
      return torch.masked_select(y, y.eq(0))

    x = torch.tensor([1., 2., 3.])
    self.runAtenTest([x], test_fn)


class TestDynamicShape(XlaTestCase):

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

  def test_index_put(self):
    xla_device = xm.xla_device()
    a = torch.tensor([1, 1, 1, 1]).to(xla_device).to(dtype=torch.float32)
    b = torch.rand(4) > 0.1
    a[b] = 10
    vset = b.sum().item()
    self.assertEqual(a.sum().item(), 10.0 * vset + (4.0 - vset))

  def test_pow_integer_types(self):
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: torch.pow(x, 2))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: torch.pow(2, x))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: torch.pow(x, x))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: x.pow_(2))
    self.runAtenTest(torch.randint(10, (2, 2)), lambda x: x.pow_(x))

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

  def test_addmm_integer_types(self):
    self.runAtenTest((torch.randint(10, (2, 3)), torch.randint(
        10, (2, 3)), torch.randint(10, (3, 3))),
                     lambda x, y, z: torch.addmm(x, y, z))

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

    # PRED can be automatically promoted in arithmetic ops.
    self.runAtenTest(c, lambda x: x + x.byte())
    # PRED cannot be automatically promoted to other dtypes in bitwise ops.
    # This is not aligned with numpy behavior which means it might change
    # in the future.
    self.assertRaises(RuntimeError, lambda: c & c.byte())
    self.assertRaises(RuntimeError, lambda: xla_c & xla_c.byte())

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

  def test_view_and_copy_(self):
    xla_device = xm.xla_device()
    x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], device='cpu')
    y = torch.tensor([0, 0, 0, 0, 0, 0], device=xla_device)
    y[::2].copy_(x[::2])
    self.assertEqual(y, [1, 0, 3, 0, 5, 0])

  def test_binaryop_order(self):
    xla_device = xm.xla_device()
    x = torch.rand(5, device=xla_device)
    y = torch.rand(5)
    self.assertEqual(x + y, y + x)


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
    SEED = 42

    xla_device = xm.xla_device()
    x = _gen_tensor(8, 1, 28, 28)
    xla_x = x.to(xla_device)

    _set_rng_seed(SEED)
    model = MNISTComparator()
    save_dir1 = xu.TmpFolder()
    mc.configure(save_dir1.name)
    model(x)

    save_dir2 = xu.TmpFolder()
    mc.configure(save_dir2.name)
    _set_rng_seed(SEED)
    xla_model = MNISTComparator().to(xla_device)
    xla_model(xla_x)

    report = mc.compare(save_dir1.name, save_dir2.name, rtol=1e-03, atol=1e-03)
    if report:
      print(report)
    self.assertEqual(len(report), 0)


class TestOpBuilder(XlaTestCase):

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


class MpDecoratorTest(XlaTestCase):

  @xtu.mp_test
  def test_mp_decorator(self):
    xla_device = xm.xla_device()
    self.assertTrue(xla_device.type == 'xla')


class TestGeneric(XlaTestCase):

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

  def test_nms(self):
    BOXES = (
        (0, 0, 3, 2),
        (3, 3, 11, 7),
        (2, 2, 5, 7),
        (7, 4, 15, 12),
    )
    SCORES = (0.9, 0.5, 0.95, 0.4)
    SCORE_THRESHOLD = 0.1
    IOU_THRESHOLD = 0.08

    xla_device = xm.xla_device()
    boxes = torch.tensor(BOXES, dtype=torch.float).to(xla_device)
    scores = torch.tensor(SCORES, dtype=torch.float).to(xla_device)
    score_threshold = torch.tensor(
        SCORE_THRESHOLD, dtype=torch.float).to(xla_device)
    iou_threshold = torch.tensor(
        IOU_THRESHOLD, dtype=torch.float).to(xla_device)

    selected_indices, num_valid = xf.nms(boxes, scores, score_threshold,
                                         iou_threshold, len(BOXES))

    self.assertEqual(selected_indices,
                     torch.tensor([2, 0, 3, 1], dtype=torch.int32))
    self.assertEqual(num_valid.item(), 3)

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


if __name__ == '__main__':
  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(met.metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
