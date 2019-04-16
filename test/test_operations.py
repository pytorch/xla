# Parse local options first, and rewrite the sys.argv[].
# We need to do that before import "common", as otherwise we get an error for
# unrecognized arguments.
import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--replicated', action='store_true')
parser.add_argument('--long_test', action='store_true')
parser.add_argument('--max_diff_count', type=int, default=25)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers
# Setup import folders.
_XLA_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.join(os.path.dirname(_XLA_FOLDER), 'test'))
sys.path.insert(0, _XLA_FOLDER)

# Normal imports section starts here.
import collections
from common_utils import TestCase, run_tests, iter_indices
import itertools
import numpy
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


def _xla_run(model, input, device='TPU'):
  if isinstance(input, (tuple, list)):
    devices = ['{}:{}'.format(device, n) for n in range(0, len(input))]
    xla_model = xm.XlaModel(
        model,
        input[0],
        num_cores=len(input),
        devices=devices,
        full_conv_precision=True)
    output_xla = xla_model(*input)
    return xm.convert_to_tensors(output_xla)
  else:
    xla_model = xm.XlaModel(model, [input], full_conv_precision=True)
    output_xla = xla_model(input)
    return output_xla[0]


class XlaTestCase(TestCase):

  def assertEqualRel(self, out, expected, rel_err=1e-2, abs_err=1e-5):
    try:
      diff_tensor = (out - expected).abs()
      max_rel_err = torch.max(out.abs(), expected.abs()) * rel_err
      # Allow higher relative differences as long as we're still below the
      # absolute error.
      max_abs_err = torch.max(max_rel_err, torch.ones_like(out) * abs_err)
      super(XlaTestCase, self).assertEqual(diff_tensor.size(),
                                           max_abs_err.size())
      if torch.le(diff_tensor, max_abs_err).min().item() == 0:
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

  def compareReplicated(self, model, inputs, xla_outputs):
    self.assertEqual(len(inputs), len(xla_outputs))
    for i, input in enumerate(inputs):
      expected = xu.as_list(model(*input))
      xla_output = xu.as_list(xla_outputs[i])
      self.assertEqual(len(expected), len(xla_output))
      for j, expected_tensor in enumerate(expected):
        self.assertEqualDbg(xla_output[j], expected_tensor)

  def compareModel(self, model, input, rel_err=0.05, abs_err=1e-4):
    xla_model = xm.XlaModel(model, [input], full_conv_precision=True)
    output_xla = xla_model(input)
    output = model(input)
    self.assertEqualRel(
        output,
        xm.convert_to_tensors(output_xla)[0],
        rel_err=rel_err,
        abs_err=abs_err)
    grad_output = _gen_tensor(*output.shape)  # random gradients
    grad_output.grad = grad_output.data
    output.backward(grad_output)
    xla_model.backward([grad_output])
    xla_updated_params = [p.grad.to_tensor() for p in xla_model.parameters()[0]]
    updated_params = [p.grad for p in model.parameters()]
    self.assertEqual(len(xla_updated_params), len(updated_params))
    for i in range(0, len(updated_params)):
      self.assertEqualRel(
          xla_updated_params[i],
          updated_params[i],
          rel_err=rel_err,
          abs_err=abs_err)


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


class TestMulAdd(XlaTestCase):

  def test(self):

    class XlaMulAdd(nn.Module):

      def forward(self, x, y):
        return x * y + y

    x = torch.rand(3, 5)
    y = torch.rand(3, 5)
    model = XlaMulAdd()
    traced_model = torch.jit.trace(model, (x, y))
    xla_model = torch_xla._XLAC.XlaModule(traced_model)
    inputs_xla = [torch_xla._XLAC.XLATensor(x), torch_xla._XLAC.XLATensor(y)]
    output_xla = xla_model((tuple(inputs_xla)))
    expected = model(x, y)
    self.assertEqualDbg(output_xla[0][0].to_tensor().data, expected.data)


class TestRelu(XlaTestCase):

  def test(self):

    class XlaRelu(nn.Module):

      def forward(self, x):
        return F.relu(x)

    x = _gen_tensor(2, 1, 4, 6)
    model = XlaRelu()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestThreshold(XlaTestCase):

  def test(self):

    class XlaThreshold(nn.Module):

      def __init__(self):
        super(XlaThreshold, self).__init__()
        self.threshold = nn.Threshold(0.4, 20)

      def forward(self, x):
        return self.threshold(x)

    x = torch.rand(4, 2)
    model = XlaThreshold()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestTranspose(XlaTestCase):

  def test(self):

    class XlaTranspose(nn.Module):

      def forward(self, x):
        return torch.t(x)

    x = torch.rand(2, 3)
    model = XlaTranspose()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestView(XlaTestCase):

  def test(self):

    class XlaView(nn.Module):

      def forward(self, x):
        return x.view(-1, 16)

    x = torch.rand(4, 8)
    model = XlaView()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestStack(XlaTestCase):

  def test(self):

    class XlaStack(nn.Module):

      def __init__(self, dim):
        super(XlaStack, self).__init__()
        self.dim = dim

      def forward(self, x, y):
        return torch.stack((x, y), self.dim)

    x = torch.rand(2, 5)
    y = torch.rand(2, 5)
    for dim in [0, 1]:
      model = XlaStack(dim)
      traced_model = torch.jit.trace(model, (x, y))
      xla_model = torch_xla._XLAC.XlaModule(traced_model, differentiate=False)
      inputs_xla = [torch_xla._XLAC.XLATensor(x), torch_xla._XLAC.XLATensor(y)]
      output_xla = xla_model((tuple(inputs_xla)))
      expected = model(x, y)
      self.assertEqualDbg(output_xla[0][0].to_tensor().data, expected.data)


class TestExpand(XlaTestCase):

  def test(self):

    class XlaExpand(nn.Module):

      def forward(self, x):
        return x.expand(2, 5)

    x = torch.rand(5)
    model = XlaExpand()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestLinear(XlaTestCase):

  def test(self):

    class XlaLinear(nn.Module):

      def __init__(self):
        super(XlaLinear, self).__init__()
        self.linear = nn.Linear(2, 5)

      def forward(self, x):
        return self.linear(x)

    x = torch.rand(4, 2)
    model = XlaLinear()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestNonContiguousTensor(XlaTestCase):

  def test(self):

    class XlaPlusSelf(nn.Module):

      def forward(self, x):
        return x + x

    x = torch.rand(3, 7)
    model = XlaPlusSelf()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)
    out_t = _xla_run(model, x.t())
    expected_t = model(x.t())
    self.assertEqualDbg(out_t.data, expected_t.data)
    self.assertEqualDbg(out_t.data, out.t().data)


class TestConstantTensor(XlaTestCase):

  def test(self):
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)

    class XplusY(nn.Module):

      def forward(self, a):
        return a + y

    class XmulY(nn.Module):

      def forward(self, a):
        return a * y

    model = XplusY()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)

    model = XmulY()
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


class TestConv(XlaTestCase):

  def test(self):

    class XlaConv(nn.Module):

      def __init__(self, stride, padding, bias):
        super(XlaConv, self).__init__()
        self.conv = nn.Conv2d(
            10, 100, 5, stride=stride, padding=padding, bias=bias)

      def forward(self, x):
        return self.conv(x)

    for stride in range(1, 4):
      for padding in range(0, 3):
        for bias in [True, False]:
          x = _gen_tensor(32, 10, 28, 28)
          model = XlaConv(stride, padding, bias)
          out = _xla_run(model, x)
          expected = model(x)
          self.assertEqualRel(out.data, expected.data)


class TestMaxPool(XlaTestCase):

  def test(self):

    class XlaMaxPool(nn.Module):

      def __init__(self, stride, padding):
        super(XlaMaxPool, self).__init__()
        self.stride = stride
        self.padding = padding

      def forward(self, x):
        return F.max_pool2d(x, 3, stride=self.stride, padding=self.padding)

    x = torch.rand(1, 64, 112, 112)
    for stride in [None, 2]:
      for padding in [0, 1]:
        model = XlaMaxPool(stride, padding)
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqualDbg(out.data, expected.data)


class TestAvgPool(XlaTestCase):

  def test(self):

    class XlaAvgPool(nn.Module):

      def __init__(self, stride, padding, count_include_pad):
        super(XlaAvgPool, self).__init__()
        self.stride = stride
        self.padding = padding
        self.count_include_pad = count_include_pad

      def forward(self, x):
        return F.avg_pool2d(x, 2, self.stride, self.padding, False,
                            self.count_include_pad)

    x = torch.rand(1, 1, 3, 3)
    for stride in [1, 2, None]:
      for padding in [0, 1]:
        for count_include_pad in [False, True]:
          model = XlaAvgPool(stride, padding, count_include_pad)
          out = _xla_run(model, x)
          expected = model(x)
          self.assertEqualDbg(out.data, expected.data)


class TestLogSoftmax(XlaTestCase):

  def test(self):

    class XlaLogSoftmax(nn.Module):

      def __init__(self, dim):
        super(XlaLogSoftmax, self).__init__()
        self.dim = dim

      def forward(self, x):
        return F.log_softmax(x, self.dim)

    x = torch.rand(5, 3, 4, 2)
    for dim in range(0, x.dim()):
      model = XlaLogSoftmax(dim)
      out = _xla_run(model, x)
      expected = model(x)
      self.assertEqualRel(out.data, expected.data, rel_err=1e-4, abs_err=1)


class TestBatchNorm(XlaTestCase):

  def test(self):

    class XlaBatchNorm(nn.Module):

      def __init__(self, training):
        super(XlaBatchNorm, self).__init__()
        if training:
          self.bn = nn.BatchNorm2d(3)
        else:
          self.bn = nn.BatchNorm2d(3, track_running_stats=False)

      def forward(self, x):
        return self.bn(x)

    x = torch.rand(14, 3, 5, 7)
    model = XlaBatchNorm(True)
    out = _xla_run(model, x)
    expected = model(x)
    self.assertEqualDbg(out.data, expected.data)


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


class TestMNIST(XlaTestCase):

  def test(self):
    batch_size = 32
    x = _gen_tensor(batch_size, 1, 28, 28)
    model = XlaMNIST()
    self.compareModel(model, x)


class TestParallelTensorMNIST(XlaTestCase):

  def test(self):
    devices = xm.get_xla_supported_devices()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=8)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    train_loader = xu.SampleGenerator(
        data=torch.zeros(batch_size, 1, 28, 28),
        target=torch.zeros(batch_size, dtype=torch.int64),
        sample_count=sample_count * len(devices))

    def loop_fn(model, loader):
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
        XlaMNIST, train_loader, loop_fn, device_ids=devices)
    model_parallel()
    if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
      print(torch_xla._XLAC._xla_metrics_report())


class TestParallelTensorResnet18(XlaTestCase):

  def test(self):
    devices = xm.get_xla_supported_devices()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=4)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    train_loader = xu.SampleGenerator(
        data=torch.zeros(batch_size, 3, 224, 224),
        target=torch.zeros(batch_size, dtype=torch.int64),
        sample_count=sample_count * len(devices))

    def loop_fn(model, loader):
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
        torchvision.models.resnet18, train_loader, loop_fn, device_ids=devices)
    model_parallel()
    if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
      print(torch_xla._XLAC._xla_metrics_report())


class AxPlusB(nn.Module):

  def __init__(self, dims=None):
    super(AxPlusB, self).__init__()
    self.ones = torch.ones(*dims)
    self.a = nn.Parameter(_gen_tensor(1, 1))
    self.b = nn.Parameter(_gen_tensor(1, 1))

  def forward(self, x):
    return x.mm(self.a) + self.ones.mm(self.b)


class SquareLoss(nn.Module):

  def __init__(self):
    super(SquareLoss, self).__init__()

  def forward(self, x, y):
    x.requires_grad = True
    y.requires_grad = True
    diff = x - y
    loss = diff.t().mm(diff)[0][0]
    return loss / x.size()[0]


class TestAxPlusB(XlaTestCase):

  def test(self):
    A = 3.11
    B = 4.09
    model = AxPlusB(dims=(1, 1))
    xla_model = xm.XlaModel(model, [_gen_tensor(1, 1)])
    optimizer = optim.SGD(xla_model.parameters_list(), lr=0.1, momentum=0.5)
    square_loss = SquareLoss()
    loss = None
    for _ in range(0, 100):
      optimizer.zero_grad()
      x = _gen_tensor(1, 1)
      target = x * A + B
      y = xla_model(x)
      loss = square_loss(y[0], target)
      loss.backward()
      xla_model.backward(y)
      optimizer.step()
    self.assertEqualRel(loss.sum(), torch.tensor(0.0))


class TestAxPlusBGen(XlaTestCase):

  def test(self):
    A = 3.11
    B = 4.09
    batch_size = 128
    gen = xu.FnDataGenerator(
        lambda x: x * A + B, batch_size, _gen_tensor, count=100)
    model = AxPlusB(dims=(batch_size, 1))
    xla_model = xm.XlaModel(model, [_gen_tensor(batch_size, 1)])
    optimizer = optim.SGD(xla_model.parameters_list(), lr=0.1, momentum=0.5)
    square_loss = SquareLoss()
    loss = None
    for x, target in gen:
      optimizer.zero_grad()
      y = xla_model(x)
      loss = square_loss(y[0], target)
      loss.backward()
      xla_model.backward(y)
      optimizer.step()
    self.assertEqualRel(loss.sum(), torch.tensor(0.0))


class TestAxPlusBGenXla(XlaTestCase):

  def test(self):
    batch_size = 128
    scaler = torch.Tensor([[1.0 / batch_size]])

    def loss_fn(x, y):
      diff = x - y
      sloss = diff.t().mm(diff)
      return sloss.mm(scaler)

    A = 3.11
    B = 4.09
    gen = xu.FnDataGenerator(
        lambda x: x * A + B, batch_size, _gen_tensor, count=100)
    model = AxPlusB(dims=(batch_size, 1))
    xla_model = xm.XlaModel(
        model, [_gen_tensor(batch_size, 1)],
        target=_gen_tensor(batch_size, 1),
        loss_fn=loss_fn,
        num_cores=1,
        devices=[':0'])
    optimizer = optim.SGD(xla_model.parameters_list(), lr=0.1, momentum=0.5)
    xla_model.train(gen, optimizer, batch_size, log_fn=None)

    def eval_fn(output, target):
      mloss = (output - target) * (output - target)
      error = torch.ones_like(mloss) * 1e-5
      count = torch.le(mloss, error).sum()
      return mloss.mean().item(), count.item()

    gen = xu.FnDataGenerator(lambda x: x * A + B, batch_size, _gen_tensor)
    accuracy = xla_model.test(gen, eval_fn, batch_size, log_fn=None)
    self.assertEqual(accuracy, 100.0)


class TestCompareAxPlusB(XlaTestCase):

  def test(self):
    batch_size = 128
    model = AxPlusB(dims=(batch_size, 1))
    x = _gen_tensor(batch_size, 1)
    self.compareModel(model, x)


class TestSum(XlaTestCase):

  def test(self):

    class XlaSum(nn.Module):

      def __init__(self, dim):
        super(XlaSum, self).__init__()
        self.dim = dim

      def forward(self, x):
        return x.sum(dim=self.dim)

    x = _gen_tensor(2, 3, 4, 6)
    for dim in range(0, x.dim()):
      model = XlaSum(dim)
      out = _xla_run(model, x)
      expected = model(x)
      self.assertEqualDbg(out.data, expected.data)


class XlaNllLoss(nn.Module):

  def __init__(self):
    super(XlaNllLoss, self).__init__()
    self.nll_loss = nn.NLLLoss()

  def forward(self, x, labels):
    return self.nll_loss(x, labels)


class TestNllLoss(TestCase):

  def test(self):
    input = _gen_tensor(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    model = XlaNllLoss()
    traced_model = torch.jit.trace(model, (input, target))
    xla_model = torch_xla._XLAC.XlaModule(traced_model)
    xla_inputs = [
        torch_xla._XLAC.XLATensor(input),
        torch_xla._XLAC.XLATensor(target)
    ]
    output_xla = xla_model((tuple(xla_inputs)))
    expected = model(input, target)
    self.assertEqual(output_xla[0][0].to_tensor().data, expected.data)


class TestLongGraphChain(XlaTestCase):

  def test(self):
    orig_x = torch.Tensor([[1, 2], [3, 4]])
    orig_y = torch.Tensor([[0.1, 0.2], [0.3, 0.4]])
    x = orig_x
    y = orig_y
    xla_x = torch_xla._XLAC.XLATensor(orig_x)
    xla_y = torch_xla._XLAC.XLATensor(orig_y)
    for i in range(0, 10000):
      x = x + 2 * y
      xla_x = xla_x.add(2, xla_y)
    self.assertEqualRel(x, xla_x.to_tensor(), rel_err=1e-3, abs_err=5)


class TestOptimizer(XlaTestCase):

  def test_inplace_add_mul(self):
    orig_x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    orig_y = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    x = orig_x
    y = orig_y
    xla_x = torch_xla._XLAC.XLATensor(orig_x)
    xla_y = torch_xla._XLAC.XLATensor(orig_y)
    self.assertEqualDbg(
        x.add_(2, y).mul_(y),
        xla_x.add_(2, xla_y).mul_(xla_y).to_tensor())
    self.assertEqualDbg(
        x.add_(y).mul_(y),
        xla_x.add_(xla_y).mul_(xla_y).to_tensor())

  def test_add_mul(self):
    orig_x = torch.Tensor([[1, 2], [3, 4]])
    orig_y = torch.Tensor([[0.1, 0.2], [0.3, 0.4]])
    x = orig_x
    y = orig_y
    xla_x = torch_xla._XLAC.XLATensor(orig_x)
    xla_y = torch_xla._XLAC.XLATensor(orig_y)
    xla_ones = torch_xla._XLAC.XLATensor(torch.ones_like(x))
    self.assertEqualDbg(x + 3 * y, xla_x.add(3, xla_y).to_tensor())
    self.assertEqualDbg(x * y, xla_x.mul(xla_y).to_tensor())
    z = (x + 9) * (y + 3)
    xla_z = xla_x.add(9, xla_ones).mul(xla_y.add(3, xla_ones))
    self.assertEqualDbg(z, xla_z.to_tensor())
    self.assertEqualDbg(x + y, (xla_x + xla_y).to_tensor())
    self.assertEqualDbg(x * y, (xla_x * xla_y).to_tensor())
    self.assertEqualDbg(x * 11.0, (xla_x * 11.0).to_tensor())
    self.assertEqualDbg(x / 3.11, (xla_x / 3.11).to_tensor())
    self.assertEqualDbg(y / x, (xla_y / xla_x).to_tensor())

  def checkSgd(self, lr, momentum, weight_decay, nsteps, do_zero_grad):
    input = _gen_tensor(4, 4, requires_grad=True)
    model = nn.Linear(4, 20)
    traced_model = torch.jit.trace(model, input)
    xla_model = torch_xla._XLAC.XlaModule(traced_model)
    input_xla = [torch_xla._XLAC.XLATensor(input)]
    xla_model((tuple(input_xla)))
    xla_optimizer = optim.SGD(
        xla_model.parameters()[0],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    output = model(input)
    grad_output = _gen_tensor(*output.shape)  # random gradients
    grad_output_xla = [torch_xla._XLAC.XLATensor(grad_output)]
    output.backward(grad_output)
    xla_model.backward((tuple(grad_output_xla)))
    if do_zero_grad:
      optimizer.zero_grad()
      xla_optimizer.zero_grad()
    for _ in range(0, nsteps):
      xla_optimizer.step()
      optimizer.step()
      xla_updated_params = [
          p.to_tensor().data for p in xla_model.parameters()[0]
      ]
      updated_params = [p.data for p in model.parameters()]
      for i in range(0, len(updated_params)):
        self.assertEqualRel(xla_updated_params[i], updated_params[i])

  def test_sgd(self):
    for weight_decay in [0, 5e-4]:
      self.checkSgd(
          lr=0.1,
          momentum=0,
          weight_decay=weight_decay,
          nsteps=1,
          do_zero_grad=True)
      self.checkSgd(
          lr=0.1,
          momentum=0,
          weight_decay=weight_decay,
          nsteps=2,
          do_zero_grad=False)
      self.checkSgd(
          lr=0.1,
          momentum=0.5,
          weight_decay=weight_decay,
          nsteps=1,
          do_zero_grad=True)
      self.checkSgd(
          lr=0.1,
          momentum=0.5,
          weight_decay=weight_decay,
          nsteps=2,
          do_zero_grad=False)


# Disabled always for now.
@unittest.skipIf(not (FLAGS.replicated and _support_replicated('TPU', 8)),
                 'Replicated (8) TPU only')
class TestReplicatedSum(XlaTestCase):

  def test(self):

    class XlaSum(nn.Module):

      def forward(self, x, y):
        return x + y

    model = XlaSum()
    for num_replicas in [2, 3, 4, 5, 6, 7, 8]:
      inputs = _random_inputs(((3, 3), (3, 3)), num_replicas=num_replicas)
      out = _xla_run(model, inputs)
      self.compareReplicated(model, inputs, out)


class TestSelect(XlaTestCase):

  def test_get_xla_tensor(self):
    x = _gen_tensor(14, 24, 8, device=xm.xla_device())
    t = x.data.cpu()
    sx = x.select(1, 12)
    tx = t.select(1, 12)
    self.assertEqual(tx, sx.data.cpu())


class TestXLATensor(XlaTestCase):

  def test_size(self):
    x = _gen_tensor(2, 1, 4, 6)
    xt_x = torch_xla._XLAC.XLATensor(x)
    rank = x.dim()
    for dim in range(-rank, rank):
      self.assertEqual(x.size(dim), xt_x.size(dim))

  def test_relu(self):
    x = _gen_tensor(2, 1, 4, 6)
    xt_x = torch_xla._XLAC.XLATensor(x)
    expected = F.relu(x)
    out = torch_xla._XLAC.relu(xt_x).to_tensor()
    self.assertEqualDbg(out.data, expected.data)

  def test_threshold(self):
    x = _gen_tensor(2, 1, 4, 6)
    xt_x = torch_xla._XLAC.XLATensor(x)
    threshold = 0.4
    value = 20
    expected = F.threshold(x, threshold, value)
    out = torch_xla._XLAC.threshold(xt_x, threshold, value).to_tensor()
    self.assertEqualDbg(out.data, expected.data)

  def test_conv2d(self):
    in_channels = 3
    out_channels = 7
    kernel_size = 5
    input = _gen_tensor(4, in_channels, 28, 28)
    weight = _gen_tensor(out_channels, in_channels, kernel_size, kernel_size)
    bias = _gen_tensor(out_channels)
    xt_input = torch_xla._XLAC.XLATensor(input)
    xt_weight = torch_xla._XLAC.XLATensor(weight)
    xt_bias = torch_xla._XLAC.XLATensor(bias)
    for stride in range(1, 4):
      for padding in range(0, 3):
        for with_bias in [True, False]:
          conv_bias = bias if with_bias else None
          expected = F.conv2d(
              input, weight, conv_bias, stride=stride, padding=padding)
          if with_bias:
            out = torch_xla._XLAC.conv2d(
                xt_input, xt_weight, xt_bias, stride=stride,
                padding=padding).to_tensor()
          else:
            out = torch_xla._XLAC.conv2d(
                xt_input, xt_weight, stride=stride,
                padding=padding).to_tensor()
          self.assertEqualRel(out.data, expected.data)

  def test_conv2d_non_square(self):
    in_channels = 3
    out_channels = 7
    kernel_size = 5
    input = _gen_tensor(4, in_channels, 28, 28)
    weight = _gen_tensor(out_channels, in_channels, kernel_size, kernel_size)
    bias = _gen_tensor(out_channels)
    xt_input = torch_xla._XLAC.XLATensor(input)
    xt_weight = torch_xla._XLAC.XLATensor(weight)
    xt_bias = torch_xla._XLAC.XLATensor(bias)
    for stride in range(1, 4):
      for padding in range(0, 3):
        for with_bias in [True, False]:
          conv_bias = bias if with_bias else None
          expected = F.conv2d(
              input,
              weight,
              conv_bias,
              stride=[stride, stride + 1],
              padding=[padding, padding + 1])
          if with_bias:
            out = torch_xla._XLAC.conv2d(
                xt_input,
                xt_weight,
                xt_bias,
                stride=[stride, stride + 1],
                padding=[padding, padding + 1]).to_tensor()
          else:
            out = torch_xla._XLAC.conv2d(
                xt_input,
                xt_weight,
                stride=[stride, stride + 1],
                padding=[padding, padding + 1]).to_tensor()
          self.assertEqualRel(out.data, expected.data)

  def test_addmm(self):
    in_channels = 32
    out_channels = 320
    labels = 50
    input = _gen_tensor(in_channels, out_channels)
    weight = _gen_tensor(out_channels, labels)
    bias = _gen_tensor(labels)
    xt_input = torch_xla._XLAC.XLATensor(input)
    xt_weight = torch_xla._XLAC.XLATensor(weight)
    xt_bias = torch_xla._XLAC.XLATensor(bias)
    out = torch.addmm(bias, input, weight)
    expected = torch_xla._XLAC.addmm(xt_bias, xt_input, xt_weight).to_tensor()
    self.assertEqualRel(out.data, expected.data)

  def test_max_pool2d(self):
    x = _gen_tensor(1, 64, 112, 112)
    xt_x = torch_xla._XLAC.XLATensor(x)
    for stride in [1, 2]:
      for padding in [0, 1]:
        expected = F.max_pool2d(x, 3, stride=stride, padding=padding)
        out = torch_xla._XLAC.max_pool2d(
            xt_x, 3, stride=stride, padding=padding).to_tensor()
        self.assertEqualRel(out.data, expected.data)

  def test_max_pool2d_non_square(self):
    x = _gen_tensor(1, 64, 112, 112)
    xt_x = torch_xla._XLAC.XLATensor(x)
    for stride in [1, 2]:
      for padding in [0, 1]:
        expected = F.max_pool2d(
            x, [3, 4],
            stride=[stride, stride + 1],
            padding=[padding, padding + 1])
        out = torch_xla._XLAC.max_pool2d(
            xt_x, [3, 4],
            stride=[stride, stride + 1],
            padding=[padding, padding + 1]).to_tensor()
        self.assertEqualRel(out.data, expected.data)

  def test_avg_pool2d(self):
    x = _gen_tensor(4, 1, 28, 28)
    xt_x = torch_xla._XLAC.XLATensor(x)
    for stride in [1, 2]:
      for padding in [0, 1]:
        for count_include_pad in [False, True]:
          expected = F.avg_pool2d(
              x,
              2,
              stride=stride,
              padding=padding,
              count_include_pad=count_include_pad)
          out = torch_xla._XLAC.avg_pool2d(
              xt_x,
              2,
              stride=stride,
              padding=padding,
              count_include_pad=count_include_pad).to_tensor()
          self.assertEqualRel(out.data, expected.data)

  def test_avg_pool2d_non_square(self):
    x = _gen_tensor(4, 1, 28, 28)
    xt_x = torch_xla._XLAC.XLATensor(x)
    for stride in [1, 2]:
      for padding in [0, 1]:
        for count_include_pad in [False, True]:
          expected = F.avg_pool2d(
              x, [4, 5],
              stride=[stride, stride + 1],
              padding=[padding, padding + 1],
              count_include_pad=count_include_pad)
          out = torch_xla._XLAC.avg_pool2d(
              xt_x, [4, 5],
              stride=[stride, stride + 1],
              padding=[padding, padding + 1],
              count_include_pad=count_include_pad).to_tensor()
          self.assertEqualRel(out.data, expected.data)

  def test_transpose(self):
    x = _gen_tensor(2, 3)
    xt_x = torch_xla._XLAC.XLATensor(x)
    expected = x.t()
    out = xt_x.t().to_tensor()
    self.assertEqualDbg(out.data, expected.data)

  def test_view(self):
    x = _gen_tensor(32, 20, 4, 4)
    xt_x = torch_xla._XLAC.XLATensor(x)
    expected = x.view(-1, 320)
    out = xt_x.view(-1, 320).to_tensor()
    self.assertEqualDbg(out.data, expected.data)

  def test_log_softmax(self):
    x = _gen_tensor(5, 3, 4, 2)
    xt_x = torch_xla._XLAC.XLATensor(x)
    for dim in range(0, x.dim()):
      expected = x.log_softmax(dim)
      out = xt_x.log_softmax(dim).to_tensor()
      self.assertEqualRel(out.data, expected.data)


class TestAtenXlaTensor(XlaTestCase):

  def test_get_xla_tensor(self):
    t = _gen_tensor(4, 2, device=xm.xla_device())
    x = torch_xla._XLAC._get_xla_tensor(t)
    self.assertEqual(t.data.cpu(), x.to_tensor())

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
    mask = torch.randint(0, 2, input.size(), dtype=torch.uint8)
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

  def test_clamp(self):
    a = torch.randn(3, 3)
    xla_a = a.to(xm.xla_device())
    b = torch.clamp(a, max=3.4)
    xla_b = torch.clamp(xla_a, max=3.4)
    self.assertEqual(b.data, xla_b.data.cpu())

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
    x = x.view(-1, 320)
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
  run_tests()
