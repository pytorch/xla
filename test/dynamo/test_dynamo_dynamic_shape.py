import unittest
import sys

import torch
import torch_xla
import torchvision

from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu


def _is_on_tpu():
  return xr.device_type() == 'TPU'


class DynamoDynamicShapeBasicTest(unittest.TestCase):

  def get_loader(self, device, sample_count, batch_size=4):
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=batch_size)
    loader = xu.SampleGenerator(
        data=(torch.randn(batch_size, 3, 224, 224, device=device),
              torch.zeros(batch_size, dtype=torch.int64, device=device)),
        sample_count=sample_count)
    return loader

  def test_dynamic_shape_basic(self):
    torch_xla.manual_seed(100)
    device = torch_xla.device()
    # model setup
    dummy_linear = torch.nn.Linear(10, 20)
    dummy_linear_xla = torch.nn.Linear(10, 20).to(device)
    dummy_linear_xla.load_state_dict(dummy_linear.state_dict())
    compiled_linear_xla = torch.compile(
        dummy_linear_xla, backend="openxla", dynamic=True)
    input = torch.randn(20, 10)
    input_xla = input.to(device)
    xm.wait_device_ops()
    met.clear_all()

    # first run
    res = dummy_linear(input)
    res_xla = compiled_linear_xla(input_xla)
    torch.allclose(res, res_xla.cpu(), atol=1e-2, rtol=1e-4)
    # torch.compile should be called once
    self.assertEqual(met.counter_value('DynamoExtractCompiledGraph'), 1)
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # second run with different input shape
    input = torch.randn(30, 10)
    input_xla = input.to(device)
    met.clear_all()
    res = dummy_linear(input)
    res_xla = compiled_linear_xla(input_xla)
    torch.allclose(res, res_xla.cpu(), atol=1e-2, rtol=1e-4)
    # torch.compile should not retrace but xla will recompile
    self.assertNotIn('DynamoExtractCompiledGraph', met.counter_names())
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

  def test_dynamic_shape_resnet18(self):
    device = torch_xla.device()

    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    loader = self.get_loader(device, sample_count, batch_size=4)
    resnet18 = torchvision.models.resnet18()
    resnet18.eval()
    device_resnet18 = torchvision.models.resnet18()
    device_resnet18.load_state_dict(resnet18.state_dict())
    device_resnet18.to(device)
    device_resnet18.eval()
    # materalize the fake data for test purpose
    xm.mark_step()
    xm.wait_device_ops()
    met.clear_all()
    dynamo_resnet18 = torch.compile(
        device_resnet18, backend='openxla', dynamic=True)
    for data, _ in loader:
      output = dynamo_resnet18(data)
      output_cpu = resnet18(data.cpu())
      # TPU has some precision issues, skipping allclose check
      if not _is_on_tpu():
        self.assertTrue(
            torch.allclose(output_cpu, output.cpu(), rtol=1e-05, atol=1e-05))

    previous_extract_compile_count = met.counter_value(
        'DynamoExtractCompiledGraph')

    loader_new_shape = self.get_loader(device, sample_count, batch_size=2)
    for data, _ in loader_new_shape:
      output_new_shape = dynamo_resnet18(data)
      output_cpu_new_shape = resnet18(data.cpu())
      # TPU has some precision issues, skipping allclose check
      if not _is_on_tpu():
        self.assertTrue(
            torch.allclose(
                output_cpu_new_shape,
                output_new_shape.cpu(),
                rtol=1e-05,
                atol=1e-05))

    self.assertEqual(
        met.counter_value('DynamoExtractCompiledGraph'),
        previous_extract_compile_count)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
