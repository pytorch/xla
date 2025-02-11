import collections
from contextlib import contextmanager
import itertools
import math
import os
import sys
import unittest
from numbers import Number

import torch
import numpy
import random
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu


def _set_rng_seed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  numpy.random.seed(seed)
  xm.set_rng_state(seed)


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


def _is_iterable(obj):
  try:
    iter(obj)
    return True
  except TypeError:
    return False


class Holder(object):
  pass


def _iter_indices(tensor):
  if tensor.dim() == 0:
    return range(0)
  if tensor.dim() == 1:
    return range(tensor.size(0))
  return itertools.product(*(range(s) for s in tensor.size()))


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

  def assertEqualRel(self,
                     out,
                     expected,
                     rel_err=1e-2,
                     abs_err=1e-5,
                     max_diff_count=0):
    try:
      out, expected = _prepare_tensors_for_diff(out, expected)
      nan_mask = torch.isnan(expected)
      self.assertTrue(torch.equal(nan_mask, torch.isnan(out)))
      out[nan_mask] = 0
      expected[nan_mask] = 0
      inf_mask = torch.isinf(expected)
      self.assertTrue(torch.equal(inf_mask, torch.isinf(out)))
      out[inf_mask] = 0
      expected[inf_mask] = 0
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
          max_diff_count=max_diff_count)
      raise

  def assertEqualDbg(self, out, expected, max_diff_count=0):
    try:
      super(XlaTestCase, self).assertEqual(out, expected)
    except:
      _dump_differences(
          expected, out, rtol=1e-8, atol=1e-8, max_diff_count=max_diff_count)
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


@contextmanager
def temporary_env(**kwargs):
  """
    Temporarily set environment variables within the context.
    
    Args:
        **kwargs: Key-value pairs representing environment variables to set.
                  For example: temporary_env(PATH='/new/path', DEBUG='1')
    """
  original_env = {}

  # Store original values and set new ones
  for key, value in kwargs.items():
    original_env[key] = os.environ.get(key, None)
    os.environ[key] = value

  try:
    yield
  finally:
    # Restore original environment variables
    for key, old_value in original_env.items():
      if old_value is None:
        # The variable was not originally set
        del os.environ[key]
      else:
        # Restore the original value
        os.environ[key] = old_value
