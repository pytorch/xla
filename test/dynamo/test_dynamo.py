import os
import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch._dynamo as dynamo
import torchvision
import unittest


class DynamoInferenceBasicTest(unittest.TestCase):

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
    torch.allclose(res_cpu, res_xla_dynamo.cpu())
    # verifiy that tracing is skipped in following runs
    met.clear_counters()
    res_xla_dynamo_2 = self.fn_simple_dynamo(xla_x, xla_y)
    self.assertNotIn('xla::add', met.counter_names())
    torch.allclose(res_cpu, res_xla_dynamo_2.cpu())
    # verify that dynamo can handle different inputs
    res_xla_dynamo_3 = self.fn_simple_dynamo(xla_x + xla_y, xla_y * 3)
    res_cpu_3 = self.fn_simple(x + y, y * 3)
    torch.allclose(res_cpu, res_xla_dynamo_3.cpu())

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
    xla_resnet18 = torchvision.models.resnet18().to(device)
    xla_resnet18.eval()
    # materalize the fake data for test purpose
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()
    for data, _ in loader:
      dynamo_resnet18 = torch.compile(
          xla_resnet18, backend='torchxla_trace_once')
      output = dynamo_resnet18(data)
      torch.allclose(resnet18(data.cpu()), output.cpu())
    # We only expect one graph for the resnet18 inference.
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], sample_count)
    self.assertEqual(
        met.metric_data('RunCachedGraphInputData')[0], sample_count)
    self.assertEqual(
        met.metric_data('RunCachedGraphOutputData')[0], sample_count)


class DynamoTrainingBasicTest(unittest.TestCase):

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

<<<<<<< HEAD
  @unittest.skip("fails with functionalization")
=======
  @dynamo.optimize('aot_torchxla_trace_once')
  def run_model_with_dynamo(self, model, data, target):
    return self.train_model(model, data, target)

>>>>>>> Re-enable dynamo tests (#4454)
  def test_simple_model(self):
    torch._dynamo.reset()
    device = xm.xla_device()
    input = torch.randn(3, 5, requires_grad=True)
    xla_input = input.detach().to(device)
    xla_input.requires_grad = True
    res_cpu = self.fn_simple(input)
    res_xla_dynamo = self.fn_simple_dynamo(xla_input)
    self.assertIn('xla::nll_loss_backward', met.counter_names())
    torch.allclose(res_cpu, res_xla_dynamo.cpu())
    torch.allclose(input.grad, xla_input.grad.cpu())
    # verifiy that tracing is skipped in following runs
    xla_input.grad = None
    met.clear_counters()
    res_xla_dynamo_2 = self.fn_simple_dynamo(xla_input)
    self.assertNotIn('xla::nll_loss_backward', met.counter_names())
    torch.allclose(res_cpu, res_xla_dynamo.cpu())
    torch.allclose(input.grad, xla_input.grad.cpu())
    # verify that dynamo can handle different inputs
    input.grad = None
    xla_input.grad = None
    res_xla_dynamo_3 = self.fn_simple_dynamo(xla_input * 2)
    res_cpu_3 = self.fn_simple(input * 2)
    torch.allclose(res_cpu, res_xla_dynamo.cpu())
    torch.allclose(input.grad, xla_input.grad.cpu())

  @unittest.skip("New failures")
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
      torch.allclose(xla_output.cpu(), cpu_output.cpu())
      # TODO(JackCaoG): Understand why `data.grad` is a pending IR starting
      # from second iteration instead of a `DeviceData`
      # torch.allclose(data.grad.cpu(), cpu_data.grad)
    # Graph 1: forward
    # Graph 2: backward
    # Graph 3: sync input for backward
    # Graph 4: sync input for backward (TODO(JackCaoG) understand why there are two graphs)
    self.assertEqual(met.metric_data('CompileTime')[0], 4)
    # We execute 3 grphs per step.
    self.assertEqual(met.metric_data('ExecuteTime')[0], sample_count * 3)
    # one for each forward and one for each backward
    self.assertEqual(
        met.metric_data('RunCachedGraphInputData')[0], sample_count * 2)
    self.assertEqual(
        met.metric_data('RunCachedGraphOutputData')[0], sample_count * 2)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
