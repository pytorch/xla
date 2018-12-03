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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import unittest


DeviceSupport = collections.namedtuple('DeviceSupport', ['num_devices'])


class Holder(object):
    pass


class FnDataGenerator(object):
    def __init__(self, func, batch_size, dim=1, count=1):
        self._func = func
        self._batch_size = batch_size
        self._dim = dim
        self._count = count
        self._emitted = 0

    def __len__(self):
        return self._count

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._emitted >= self._count:
            raise StopIteration
        data = torch.randn(self._batch_size, self._dim)
        target = self._func(data)
        self._emitted += 1
        return data, target


def _get_device_support(devname):
    assert devname in ['TPU', 'CPU']
    # If the Cloud TPU config file is present, we support TPUs.
    if (os.path.isfile(os.path.join(os.environ['HOME'], '.pytorch_tpu.conf')) or
            os.environ.get('XRT_TPU_CONFIG', None)):
        if devname == 'TPU':
            return DeviceSupport(num_devices=int(
                os.environ.get('TPU_NUM_DEVICES', 8)))
        return DeviceSupport(num_devices=int(
            os.environ.get('CPU_NUM_DEVICES', 1)))
    xrt = os.environ.get('XLA_USE_XRT', None)
    if xrt is None or int(xrt) == 0:
        xla_platform = os.environ.get('XLA_PLATFORM', None)
        if xla_platform == devname:
            return DeviceSupport(num_devices=1)
    else:
        xrt_devmap = os.environ.get('XRT_DEVICE_MAP', None)
        if xrt_devmap is None:
            return None
        num_devices = 0
        for dev_spec in xrt_devmap.split('|'):
            dev_parts = dev_spec.split(';')
            if dev_parts[0].startswith(devname):
                num_devices += 1
        if num_devices > 0:
            return DeviceSupport(num_devices=num_devices)
    return None


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
            replica_inputs.append(torch.randn(*shape))
        random_tensors.append(tuple(replica_inputs))
    return tuple(random_tensors)


def _random_like(tensor_list):
    random_tensors = []
    for o in tensor_list:
        if o.dtype == torch.float32 or o.dtype == torch.float64:
            random_tensors += [torch.randn(*o.shape, dtype=o.dtype)]
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
        if diff > min(r, atol):
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
        print('\nmax_diff={}\tmax_rel={}\tindex={}'.format(
            env.max_diff, env.max_rel, env.max_index))


def _xla_run(model, input, device='TPU'):
    if isinstance(input, (tuple, list)):
        devices = ['{}:{}'.format(device, n) for n in range(0, len(input))]
        xla_model = xm.XlaModel(model, input[0], num_cores=len(input),
                                devices=devices, full_conv_precision=True)
        output_xla = xla_model(*input)
        output = []
        for xla_replica_outputs in output_xla:
            replica_outputs = []
            for o in xu.as_list(xla_replica_outputs):
                replica_outputs.append(o.to_tensor())
            output.append(tuple(replica_outputs))
        return tuple(output)
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
            max_abs_err = torch.max(
                max_rel_err, torch.ones_like(out) * abs_err)
            super(XlaTestCase, self).assertEqual(diff_tensor.size(),
                                                 max_abs_err.size())
            if torch.le(diff_tensor, max_abs_err).min().item() == 0:
                self.fail('Relative error higher than the maximum tolerance')
        except:
            _dump_differences(out, expected, rtol=rel_err, atol=abs_err,
                              max_diff_count=FLAGS.max_diff_count)
            raise

    def assertEqualDbg(self, out, expected):
        try:
            super(XlaTestCase, self).assertEqual(out, expected)
        except:
            _dump_differences(out, expected, rtol=1e-8, atol=1e-8,
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

        x = torch.randn(2, 1, 4, 6)
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
            xla_model = torch_xla._XLAC.XlaModule(
                traced_model, differentiate=False)
            inputs_xla = [torch_xla._XLAC.XLATensor(x), torch_xla._XLAC.XLATensor(y)]
            output_xla = xla_model((tuple(inputs_xla)))
            expected = model(x, y)
            self.assertEqualDbg(output_xla[0][0].to_tensor().data,
                                expected.data)


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
                self.conv = nn.Conv2d(10, 100, 5, stride=stride,
                                      padding=padding, bias=bias)

            def forward(self, x):
                return self.conv(x)

        for stride in range(1, 4):
            for padding in range(0, 3):
                for bias in [True, False]:
                    x = torch.randn(32, 10, 28, 28)
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
                return F.max_pool2d(x, 3, stride=self.stride,
                                    padding=self.padding)

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
                return F.avg_pool2d(x, 2, self.stride, self.padding, False, self.count_include_pad)

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
            self.assertEqualRel(out.data, expected.data,
                                rel_err=1e-4, abs_err=1)


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
        x = torch.randn(batch_size, 1, 28, 28)
        model = XlaMNIST()
        out = _xla_run(model, x)
        expected = model(x)
        self.assertEqualDbg(out.data, expected.data)


class AxPlusB(nn.Module):
    def __init__(self, dims=None):
        super(AxPlusB, self).__init__()
        self.ones = torch.ones(*dims)
        self.a = nn.Parameter(torch.randn(1, 1))
        self.b = nn.Parameter(torch.randn(1, 1))

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
        xla_model = xm.XlaModel(model, [torch.randn(1, 1)])
        optimizer = optim.SGD(xla_model.parameters_list(),
                              lr=0.1, momentum=0.5)
        square_loss = SquareLoss()
        loss = None
        for _ in range(0, 100):
            optimizer.zero_grad()
            x = torch.randn(1, 1)
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
        gen = FnDataGenerator(lambda x: x * A + B, batch_size, count=100)
        model = AxPlusB(dims=(batch_size, 1))
        xla_model = xm.XlaModel(model, [torch.randn(batch_size, 1)])
        optimizer = optim.SGD(xla_model.parameters_list(),
                              lr=0.1, momentum=0.5)
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
        gen = FnDataGenerator(lambda x: x * A + B, batch_size, count=100)
        model = AxPlusB(dims=(batch_size, 1))
        xla_model = xm.XlaModel(model, [torch.randn(batch_size, 1)],
                                target=torch.randn(batch_size, 1),
                                loss_fn=loss_fn, num_cores=1, devices=[':0'])
        optimizer = optim.SGD(xla_model.parameters_list(),
                              lr=0.1, momentum=0.5)
        xla_model.train(gen, optimizer, batch_size, log_fn=None)

        def eval_fn(output, target):
            mloss = (output - target) * (output - target)
            error = torch.ones_like(mloss) * 1e-5
            count = torch.le(mloss, error).sum()
            return mloss.mean().item(), count.item()

        gen = FnDataGenerator(lambda x: x * A + B, batch_size)
        accuracy = xla_model.test(gen, eval_fn, batch_size, log_fn=None)
        self.assertEqual(accuracy, 100.0)


class TestSum(XlaTestCase):
    def test(self):

        class XlaSum(nn.Module):
            def __init__(self, dim):
                super(XlaSum, self).__init__()
                self.dim = dim

            def forward(self, x):
                return x.sum(dim=self.dim)

        x = torch.randn(2, 3, 4, 6)
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
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        model = XlaNllLoss()
        traced_model = torch.jit.trace(model, (input, target))
        xla_model = torch_xla._XLAC.XlaModule(traced_model)
        xla_inputs = [torch_xla._XLAC.XLATensor(
            input), torch_xla._XLAC.XLATensor(target)]
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


class TestGradients(XlaTestCase):
    def checkGrad(self, model, inputs, grad_outputs='random', xla=True,
                  rel_err=1e-2, abs_err=1e-5):
        # Trace and symbolically differentiate
        traced_model = torch.jit.trace(model, *inputs)
        fwd = traced_model._get_method('forward')
        xm.forward_passes(fwd.graph)

        inputs_params = inputs + list(model.parameters())
        inputs_params_buffers = inputs + list(fwd.params())

        gradient = torch._C._jit_differentiate(fwd.graph)
        xm.forward_passes(gradient.f)
        xm.backward_passes(gradient.df)

        ##############################################################
        # Run forward and backwarg graphs via jit interpreter
        exec_f = torch._C.GraphExecutor(gradient.f, False)
        exec_df = torch._C.GraphExecutor(gradient.df, False)

        # forward function
        raw_outputs = exec_f(*inputs_params_buffers)
        raw_outputs = xu.as_list(raw_outputs)
        intermediate_outputs = [raw_output for raw_output in raw_outputs[gradient.f_real_outputs:]
                                if raw_output.dtype == torch.float32]
        outputs = raw_outputs[:gradient.f_real_outputs]

        if grad_outputs == 'random':
            grad_outputs = _random_like(
                outputs) + _zeros_like(intermediate_outputs)

        raw_grad_outputs = []
        raw_grad_outputs += grad_outputs
        raw_grad_outputs += [inputs_params_buffers[i]
                             for i in gradient.df_input_captured_inputs]
        raw_grad_outputs += [raw_outputs[i]
                             for i in gradient.df_input_captured_outputs]

        grad_inputs = exec_df(*raw_grad_outputs)
        grad_inputs = xu.as_list(grad_inputs)

        ##############################################################
        # backward with XLA
        if xla:
            xla_model = torch_xla._XLAC.XlaModule(
                traced_model, use_full_conv_precision=True)
            inputs_xla = [torch_xla._XLAC.XLATensor(input) for input in inputs]
            xla_model((tuple(inputs_xla)))
            grads_output_xla = [torch_xla._XLAC.XLATensor(grad_output)
                                for grad_output in grad_outputs[:gradient.f_real_outputs]]
            xla_model.backward((tuple(grads_output_xla)))
            grad_inputs_xla = [input_xla.grad.to_tensor()
                               for input_xla in inputs_xla]
            grad_inputs_xla.extend([p.grad.to_tensor()
                                    for p in xla_model.parameters()[0]])
        ##############################################################
        # forward + backward with regular autograd / torch
        outputs_gt = model(*inputs)
        outputs_gt = xu.as_list(outputs_gt)
        grad_inputs_gt = torch.autograd.grad(outputs_gt,
                                             inputs_params,
                                             grad_outputs,
                                             only_inputs=True)
        for out_jit, out_autograd in zip(outputs, outputs_gt):
            self.assertEqualRel(out_jit, out_autograd, rel_err=rel_err,
                                abs_err=abs_err)

        for grad_input_jit, grad_input_autograd in zip(grad_inputs, grad_inputs_gt):
            self.assertEqualRel(grad_input_jit, grad_input_autograd,
                                rel_err=rel_err, abs_err=abs_err)

        # TODO: test buffers as well (running_mean, etc.)
        if xla:
            for i, (grad_input_jit, grad_input_xla) in enumerate(zip(grad_inputs,
                                                                     grad_inputs_xla)):
                self.assertEqualRel(
                    grad_input_jit, grad_input_xla, rel_err, abs_err)

    def test_avgpool(self):
        class AvgPoolGrad(nn.Module):
            def __init__(self, stride, padding, count_include_pad):
                super(AvgPoolGrad, self).__init__()
                self.stride = stride
                self.padding = padding
                self.count_include_pad = count_include_pad

            def forward(self, x):
                return F.avg_pool2d(x, 2, self.stride, self.padding, False,
                                    self.count_include_pad)

        for stride in [1, 2]:
            for padding in [0, 1]:
                for count_include_pad in [False, True]:
                    model = AvgPoolGrad(stride, padding, count_include_pad)
                    inputs = [torch.randn(4, 1, 28, 28, requires_grad=True)]
                    self.checkGrad(model, inputs, xla=True)

    def test_threshold(self):
        class ThresholdPoolGrad(nn.Module):
            def __init__(self):
                super(ThresholdPoolGrad, self).__init__()
                self.threshold = nn.Threshold(0.4, 20)

            def forward(self, x):
                return self.threshold(x)

        model = ThresholdPoolGrad()
        inputs = [torch.randn(4, 2, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    def test_maxpool(self):
        class MaxPoolGrad(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 2)

        model = MaxPoolGrad()
        inputs = [torch.randn(4, 1, 28, 28, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    def test_tanh(self):
        class TanhGrad(nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        model = TanhGrad()
        inputs = [torch.randn(4, 2, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    def test_sigmoid(self):
        class SigmoidGrad(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        model = SigmoidGrad()
        inputs = [torch.randn(4, 2, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True, rel_err=1e-2, abs_err=1e-2)

    @unittest.skip('differentiation of prim::ListUnpack is not supported, or it is missing necessary type information')
    def test_chunk(self):
        class ChunkGrad(nn.Module):
            def forward(self, x):
                return x.chunk(2, 1)

        model = ChunkGrad()
        inputs = [torch.randn(4, 4, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    @unittest.skip('bool value of Tensor with more than one value is ambiguous')
    def test_lstm_cell(self):
        class LSTMCellGrad(nn.Module):
            def __init__(self):
                super(LSTMCellGrad, self).__init__()
                self.i2h = nn.Linear(3, 8)
                self.h2h = nn.Linear(2, 8)

            def forward(self, x, hx, cx):
                gates = self.i2h(x) + self.h2h(hx)

                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = torch.tanh(cellgate)
                outgate = torch.sigmoid(outgate)

                cy = (forgetgate * cx) + (ingate * cellgate)
                hy = outgate * torch.tanh(cy)
                return hy, cy

        model = LSTMCellGrad()
        inputs = [torch.randn(4, 3, requires_grad=True),
                  torch.randn(4, 2, requires_grad=True),
                  torch.randn(4, 2, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    def test_conv2d(self):
        if FLAGS.long_test:
            config = [
                [1, 7, 15, 32],   # ichans
                [1, 4, 21, 32],   # ochans
                [1, 2, 3, 5],     # size
                [1, 2],           # stride
                [0, 1],           # padding
                [True, False],    # bias
            ]
        else:
            config = [
                [1, 5],        # ichans
                [1, 4],        # ochans
                [1, 3],        # size
                [1],           # stride
                [0],           # padding
                [False],       # bias
            ]
        for ichans, ochans, size, stride, padding, bias in (
                itertools.product(*config)):
            # TODO: dilation, groups, transpose
            model = nn.Conv2d(ichans, ochans, size, stride, padding, bias=bias)
            inputs = [torch.randn(4, ichans, 28, 28, requires_grad=True)]
            self.checkGrad(model, inputs, xla=True, abs_err=1e-3)

    def test_batchnorm2d(self):
        for chans in [1, 15, 32]:
            for eps in [1e-5, 1e-3, 1e-2]:
                # TODO: momentum, training, affine
                model = nn.BatchNorm2d(chans, eps=eps)
                inputs = [torch.randn(4, chans, 28, 28, requires_grad=True)]
                self.checkGrad(model, inputs, xla=True)

    def test_logsoftmax(self):
        for dim in [0, 1]:  # todo test 3d as well
            for batch in [1, 3, 4]:
                class LSMGrad(nn.Module):
                    def forward(self, x):
                        return F.log_softmax(x, dim)

                model = LSMGrad()
                inputs = [torch.randn(batch, 9, requires_grad=True)]
                self.checkGrad(model, inputs, xla=True)

    def test_nll_loss(self):
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        model = XlaNllLoss()
        traced_model = torch.jit.trace(model, (input, target))
        xla_model = torch_xla._XLAC.XlaModule(traced_model)
        xla_inputs = [torch_xla._XLAC.XLATensor(
            input), torch_xla._XLAC.XLATensor(target)]
        output_xla = xla_model((tuple(xla_inputs)))
        xla_model.backward(*output_xla)
        output = model(input, target)
        output.backward()
        self.assertEqual(input.grad.data, xla_inputs[0].grad.data.to_tensor())

    def test_mnist(self):
        model = XlaMNIST()
        inputs = [torch.randn(4, 1, 28, 28, requires_grad=True)]
        self.checkGrad(model, inputs, xla=True)

    @unittest.skip('Disable until we figure out the precision issue')
    def test_resnet(self):
        import torchvision
        model = torchvision.models.resnet50()
        inputs = [torch.randn(4, 3, 224, 224, requires_grad=True)]
        self.checkGrad(model, inputs, xla=False)


class TestOptimizer(XlaTestCase):
    def test_inplace_add_mul(self):
        orig_x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        orig_y = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        x = orig_x
        y = orig_y
        xla_x = torch_xla._XLAC.XLATensor(orig_x)
        xla_y = torch_xla._XLAC.XLATensor(orig_y)
        self.assertEqualDbg(x.add_(2, y).mul_(y),
                            xla_x.add_(2, xla_y).mul_(xla_y).to_tensor())
        self.assertEqualDbg(x.add_(y).mul_(y),
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
        input = torch.randn(4, 4, requires_grad=True)
        model = nn.Linear(4, 20)
        traced_model = torch.jit.trace(model, input)
        xla_model = torch_xla._XLAC.XlaModule(
            traced_model, use_full_conv_precision=True)
        input_xla = [torch_xla._XLAC.XLATensor(input)]
        xla_model((tuple(input_xla)))
        xla_optimizer = optim.SGD(xla_model.parameters()[0], lr=lr,
                                  momentum=momentum, weight_decay=weight_decay)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
        output = model(input)
        grad_output = torch.randn(*output.shape)  # random gradients
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
                p.to_tensor().data for p in xla_model.parameters()[0]]
            updated_params = [p.data for p in model.parameters()]
            for i in range(0, len(updated_params)):
                self.assertEqualRel(xla_updated_params[i], updated_params[i])

    def test_sgd(self):
        for weight_decay in [0, 5e-4]:
            self.checkSgd(lr=0.1, momentum=0, weight_decay=weight_decay,
                          nsteps=1, do_zero_grad=True)
            self.checkSgd(lr=0.1, momentum=0, weight_decay=weight_decay, nsteps=2,
                          do_zero_grad=False)
            self.checkSgd(lr=0.1, momentum=0.5, weight_decay=weight_decay, nsteps=1,
                          do_zero_grad=True)
            self.checkSgd(lr=0.1, momentum=0.5, weight_decay=weight_decay, nsteps=2,
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
            inputs = _random_inputs(
                ((3, 3), (3, 3)), num_replicas=num_replicas)
            out = _xla_run(model, inputs)
            self.compareReplicated(model, inputs, out)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    run_tests()
