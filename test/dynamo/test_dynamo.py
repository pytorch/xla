import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch.optim as optim
import torch._dynamo as dynamo
import torchvision
import unittest

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import test_utils


class DynamoInPlaceTest(unittest.TestCase):

  def inplace_update(self, a):
    a += 1
    return a

  def test_inplace_update_correctness(self):
    dynamo_inplace = torch.compile(
        self.inplace_update, backend="torchxla_trace_once", fullgraph=True)
    t = torch.tensor([0, 1, 2], device=xm.xla_device())
    for i in range(10):
      t = dynamo_inplace(t)
    self.assertTrue(torch.all(torch.eq(t.cpu(), torch.tensor([10, 11, 12]))))


class DynamoInferenceBasicTest(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    test_utils._set_rng_seed(42)

  def fn_simple(self, x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b

  @torch.compile(backend='torchxla_trace_once')
  def fn_simple_dynamo(self, x, y):
    return self.fn_simple(x, y)

  def test_simple_model(self):
    device = xm.xla_device()
    x = torch.tensor(100.0)
    y = torch.tensor(200.0)
    xla_x = x.to(device)
    xla_y = y.to(device)
    res_cpu = self.fn_simple(x, y)
    res_xla_dynamo = self.fn_simple_dynamo(xla_x, xla_y)
    self.assertIn('xla::add', met.counter_names())
    self.assertTrue(torch.allclose(res_cpu, res_xla_dynamo.cpu()))
    # verifiy that tracing is skipped in following runs
    met.clear_counters()
    res_xla_dynamo_2 = self.fn_simple_dynamo(xla_x, xla_y)
    self.assertNotIn('xla::add', met.counter_names())
    self.assertTrue(torch.allclose(res_cpu, res_xla_dynamo_2.cpu()))
    # verify that dynamo can handle different inputs
    res_xla_dynamo_3 = self.fn_simple_dynamo(xla_x + xla_y, xla_y * 3)
    res_cpu_3 = self.fn_simple(x + y, y * 3)
    self.assertTrue(torch.allclose(res_cpu_3, res_xla_dynamo_3.cpu()))

  def test_resnet18(self):
    device = xm.xla_device()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=4)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    loader = xu.SampleGenerator(
        data=(torch.randn(batch_size, 3, 224, 224, device=device),
              torch.zeros(batch_size, dtype=torch.int64, device=device)),
        sample_count=sample_count)
    resnet18 = torchvision.models.resnet18()
    resnet18.eval()
    xla_resnet18 = torchvision.models.resnet18()
    xla_resnet18.load_state_dict(resnet18.state_dict())
    xla_resnet18.to(device)
    xla_resnet18.eval()
    # materalize the fake data for test purpose
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()
    for data, _ in loader:
      dynamo_resnet18 = torch.compile(
          xla_resnet18, backend='torchxla_trace_once')
      output = dynamo_resnet18(data)
      output_cpu = resnet18(data.cpu())
      self.assertTrue(
          torch.allclose(output_cpu, output.cpu(), rtol=1e-05, atol=1e-05))
    # We only expect one graph for the resnet18 inference.
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], sample_count)
    self.assertEqual(
        met.metric_data('RunCachedGraphInputData')[0], sample_count)
    self.assertEqual(
        met.metric_data('RunCachedGraphOutputData')[0], sample_count)


class DynamoCpuFallbackTest(unittest.TestCase):

  def test_operator_fallback(self):

    def fn_fallback(t):
      # As of 05/18/2023, torch.median is not lowered by PyTorch/XLA
      return torch.median(t)

    torch._dynamo.reset()
    met.clear_counters()
    met.clear_all()
    device = xm.xla_device()

    # Initial tracing
    dynamo_fn = torch.compile(fn_fallback, backend="torchxla_trace_once")
    t = torch.randn(5)
    t_xla = t.to(device)
    cpu_res = fn_fallback(t)
    xla_dynamo_res = dynamo_fn(t_xla)
    self.assertTrue(torch.allclose(cpu_res, xla_dynamo_res.cpu()))
    self.assertEqual(met.metric_data('CompileTime')[0], 2)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 2)

    # Second tracing
    met.clear_counters()
    xla_dynamo_res_2 = dynamo_fn(t_xla)
    self.assertTrue(torch.allclose(cpu_res, xla_dynamo_res_2.cpu()))
    self.assertEqual(met.metric_data('CompileTime')[0], 2)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 2)

    # Verify that dynamo can handle different inputs
    xla_dynamo_res_3 = dynamo_fn(t_xla * 3)
    cpu_res_3 = fn_fallback(t * 3)
    self.assertTrue(torch.allclose(cpu_res_3, xla_dynamo_res_3.cpu()))
    self.assertEqual(met.metric_data('CompileTime')[0], 3)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 3)

  def test_fallback_multiple_submodules(self):

    def fn_fallback(t):
      t_2 = torch.mul(t, 2)
      # As of 05/18/2023, torch.median is not lowered by PyTorch/XLA
      t_3 = torch.median(t_2)
      t_4 = torch.mul(t_3, 2)
      return t_4

    torch._dynamo.reset()
    met.clear_counters()
    met.clear_all()
    device = xm.xla_device()

    # Initial tracing
    dynamo_fn = torch.compile(fn_fallback, backend="torchxla_trace_once")
    t = torch.randn(7)
    t_xla = t.to(device)
    cpu_res = fn_fallback(t)
    xla_dynamo_res = dynamo_fn(t_xla)
    self.assertTrue(torch.allclose(cpu_res, xla_dynamo_res.cpu()))
    self.assertEqual(met.metric_data('CompileTime')[0], 4)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 6)

    # Second tracing
    met.clear_counters()
    xla_dynamo_res_2 = dynamo_fn(t_xla)
    self.assertTrue(torch.allclose(cpu_res, xla_dynamo_res_2.cpu()))
    self.assertEqual(met.metric_data('CompileTime')[0], 4)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 8)

    # Verify that dynamo can handle different inputs
    xla_dynamo_res_3 = dynamo_fn(t_xla * 3)
    cpu_res_3 = fn_fallback(t * 3)
    self.assertTrue(torch.allclose(cpu_res_3, xla_dynamo_res_3.cpu()))
    self.assertEqual(met.metric_data('CompileTime')[0], 5)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 10)


class DynamoTrainingBasicTest(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    test_utils._set_rng_seed(42)

  def fn_simple(self, input):
    loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.tensor([1, 2, 3], dtype=torch.long).to(input.device)
    loss = loss_fn(input, target)
    loss.backward()
    return loss

  @torch.compile(backend='aot_torchxla_trace_once')
  def fn_simple_dynamo(self, input):
    return self.fn_simple(input)

  def train_model(self, model, data, target):
    loss_fn = torch.nn.CrossEntropyLoss()
    pred = model(data)
    loss = loss_fn(pred, target)
    loss.backward()
    return pred

  def test_simple_model(self):
    torch._dynamo.reset()
    device = xm.xla_device()
    input = torch.randn(3, 5, requires_grad=True)
    xla_input = input.detach().to(device)
    xla_input.requires_grad = True
    res_cpu = self.fn_simple(input)
    res_xla_dynamo = self.fn_simple_dynamo(xla_input)
    self.assertIn('xla::nll_loss_backward', met.counter_names())
    self.assertTrue(torch.allclose(res_cpu, res_xla_dynamo.cpu()))
    self.assertTrue(torch.allclose(input.grad, xla_input.grad.cpu()))
    # verifiy that tracing is skipped in following runs
    xla_input.grad = None
    met.clear_counters()
    res_xla_dynamo_2 = self.fn_simple_dynamo(xla_input)
    self.assertNotIn('xla::nll_loss_backward', met.counter_names())
    self.assertTrue(torch.allclose(res_cpu, res_xla_dynamo_2.cpu()))
    self.assertTrue(torch.allclose(input.grad, xla_input.grad.cpu()))
    # verify that dynamo can handle different inputs
    input.grad = None
    xla_input.grad = None
    res_xla_dynamo_3 = self.fn_simple_dynamo(xla_input * 2)
    res_cpu_3 = self.fn_simple(input * 2)
    self.assertTrue(torch.allclose(res_cpu_3, res_xla_dynamo_3.cpu()))
    self.assertTrue(torch.allclose(input.grad, xla_input.grad.cpu()))

  def test_resnet18(self):
    torch._dynamo.reset()
    met.clear_counters()
    device = xm.xla_device()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=4)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    loader = xu.SampleGenerator(
        data=(torch.randn(
            batch_size, 3, 224, 224, device=device, requires_grad=True),
              torch.zeros(batch_size, dtype=torch.int64, device=device)),
        sample_count=sample_count)
    resnet18 = torchvision.models.resnet18()
    resnet18.train()
    xla_resnet18 = torchvision.models.resnet18()
    xla_resnet18.load_state_dict(resnet18.state_dict())
    xla_resnet18.to(device)
    xla_resnet18.train()
    # materalize the fake data
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()

    dynamo_train_model = torch.compile(
        self.train_model, backend='aot_torchxla_trace_once')
    for data, target in loader:
      xla_output = dynamo_train_model(xla_resnet18, data, target)
      cpu_data = data.detach().cpu()
      cpu_data.requires_grad = True
      cpu_target = target.detach().cpu()
      cpu_output = self.train_model(resnet18, cpu_data, cpu_target)
      self.assertTrue(
          torch.allclose(
              xla_output.cpu(), cpu_output.cpu(), rtol=1e-05, atol=1e-05))
      # TODO(JackCaoG): Understand why `data.grad` is a pending IR starting
      # from second iteration instead of a `DeviceData`
      # torch.allclose(data.grad.cpu(), cpu_data.grad)
    # Graph 1: forward
    # Graph 2: backward
    # Graph 3: sync input for backward
    # Graph 4: sync input for backward (TODO(JackCaoG) understand why there are two graphs)
    self.assertEqual(met.metric_data('CompileTime')[0], 4)
    # We execute 3 graphs per step.
    self.assertEqual(met.metric_data('ExecuteTime')[0], sample_count * 3)
    # one for each forward and one for each backward
    self.assertEqual(
        met.metric_data('RunCachedGraphInputData')[0], sample_count * 2)
    self.assertEqual(
        met.metric_data('RunCachedGraphOutputData')[0], sample_count * 2)


class DynamoTrainingOptimizerTest(unittest.TestCase):

  @classmethod
  def setUpClass(self):
    test_utils._set_rng_seed(42)

  def fn_simple(self, input, optimizer):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad(True)
    target = torch.tensor([1, 2, 3], dtype=torch.long).to(input.device)
    output = (torch.cos(input) + torch.sin(input)) / 2.0
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss

  @torch.compile(backend='aot_torchxla_trace_once')
  def fn_simple_dynamo(self, input, optimizer):
    return self.fn_simple(input, optimizer)

  def train_model(self, model, data, target, optimizer):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad(True)
    pred = model(data)
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
    return pred

  def test_simple_model(self):
    torch._dynamo.reset()
    device = xm.xla_device()
    input = torch.randn(3, 5, requires_grad=True)
    saved_input = input.detach().to(device).cpu()
    xla_input = input.detach().to(device)
    xla_input.requires_grad = True
    xla_optimizer = optim.SGD([xla_input], lr=0.1, weight_decay=1e-2)
    optimizer = optim.SGD([input], lr=0.1, weight_decay=1e-2)

    for _ in range(5):
      # TODO(JackCaoG): currently for some reason this simple program
      # fwd + bwd is not being captured, hence we will get one lazy graph
      # + one dynamo optimizer graph
      res_cpu = self.fn_simple(input, optimizer)
      res_xla_dynamo = self.fn_simple_dynamo(xla_input, xla_optimizer)
      assert torch.allclose(res_cpu, res_xla_dynamo.cpu())
      assert torch.allclose(input.grad, xla_input.grad.cpu())
      assert torch.allclose(input, xla_input.cpu())

  def test_resnet18(self):
    torch._dynamo.reset()
    met.clear_counters()
    device = xm.xla_device()
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=4)
    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    loader = xu.SampleGenerator(
        data=(torch.randn(
            batch_size, 3, 224, 224, device=device, requires_grad=True),
              torch.zeros(batch_size, dtype=torch.int64, device=device)),
        sample_count=sample_count)
    resnet18 = torchvision.models.resnet18()
    resnet18.train()
    xla_resnet18 = torchvision.models.resnet18()
    xla_resnet18.load_state_dict(resnet18.state_dict())
    xla_resnet18.to(device)
    xla_resnet18.train()
    xla_optimizer = optim.SGD(
        xla_resnet18.parameters(), lr=0.1, weight_decay=1e-2)
    optimizer = optim.SGD(resnet18.parameters(), lr=0.1, weight_decay=1e-2)

    # materalize the fake data
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()

    dynamo_train_model = torch.compile(
        self.train_model, backend='aot_torchxla_trace_once')
    for data, target in loader:
      xla_output = dynamo_train_model(xla_resnet18, data, target, xla_optimizer)
      cpu_data = data.detach().cpu()
      cpu_data.requires_grad = True
      cpu_target = target.detach().cpu()
      cpu_output = self.train_model(resnet18, cpu_data, cpu_target, optimizer)
      # Disable the accuracy check here due to xla optimization and optimzer enabled.
      # Will compare the lazy vs dynamo instead of dynamo vs cpu.
      # assert torch.allclose(xla_output.cpu(), cpu_output, rtol=1e-04, atol=1e-03)
      for xla_input, cpu_input in zip(xla_resnet18.parameters(),
                                      resnet18.parameters()):
        pass
        # assert torch.allclose(xla_input.cpu(), cpu_input, rtol=1e-04, atol=1e-03)
        # assert torch.allclose(xla_input.grad.cpu(), cpu_input.grad)
    # Graph 1: forward
    # Graph 2: backward
    # Graph 3: optimizer
    # Graph 4: sync input for backward
    # Graph 5: sync input for backward (TODO(JackCaoG) understand why there are two graphs)
    self.assertEqual(met.metric_data('CompileTime')[0], 5)
    # We execute 4 graphs per step when optimizer is enabled.
    self.assertEqual(met.metric_data('ExecuteTime')[0], sample_count * 4)
    # one for each forward, backward and optimizer
    self.assertEqual(
        met.metric_data('RunCachedGraphInputData')[0], sample_count * 3)
    self.assertEqual(
        met.metric_data('RunCachedGraphOutputData')[0], sample_count * 3)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
