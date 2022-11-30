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


class DynamoBasicTest(unittest.TestCase):

  def fn_simple(self, x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b

  @dynamo.optimize('torchxla_trace_once')
  def fn_simple_dynamo(self, x, y):
    return self.fn_simple(x, y)

  @dynamo.optimize('torchxla_trace_once')
  def run_model_with_dynamo(self, model, data):
    return model(data)

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
    for data, _ in loader:
      output = self.run_model_with_dynamo(xla_resnet18, data)
      torch.allclose(resnet18(data.cpu()), output.cpu())
    # One graph for initial input data materialization. Another grpah for the
    # real model code.
    self.assertEqual(met.metric_data('CompileTime')[0], 2)
    self.assertEqual(met.metric_data('ExecuteTime')[0], sample_count + 2)
    self.assertEqual(
        met.metric_data('RunCachedGraphInputData')[0], sample_count)
    self.assertEqual(
        met.metric_data('RunCachedGraphOutputData')[0], sample_count)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
