import sys
import os
example_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(
        sys.argv[0])))) + "/examples"
sys.path.append(example_folder)
from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel

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

  def _get_loader(self, device, sample_count, batch_size=4):
    batch_size = xu.getenv_as('BATCH_SIZE', int, defval=batch_size)
    loader = xu.SampleGenerator(
        data=(torch.randn(batch_size, 3, 224, 224, device=device),
              torch.zeros(batch_size, dtype=torch.int64, device=device)),
        sample_count=sample_count)
    return loader

  def _get_linear_and_input(self, in_dim: int, out_dum: int, batch_dim: int,
                            device: torch.device):
    dummy_linear = torch.nn.Linear(in_dim, out_dum)
    dummy_linear_xla = torch.nn.Linear(in_dim, out_dum).to(device)
    dummy_linear_xla.load_state_dict(dummy_linear.state_dict())
    input = torch.randn(batch_dim, in_dim)
    input_xla = input.to(device)
    return (dummy_linear, dummy_linear_xla, input, input_xla)

  def test_dynamic_shape_basic(self):
    torch_xla.manual_seed(100)
    device = torch_xla.device()
    # model setup
    dummy_linear, dummy_linear_xla, input, input_xla = self._get_linear_and_input(
        10, 20, 20, device)
    compiled_linear_xla = torch.compile(
        dummy_linear_xla, backend="openxla", dynamic=True)
    xm.wait_device_ops()
    met.clear_all()

    # first run
    res = dummy_linear(input)
    res_xla = compiled_linear_xla(input_xla)
    # TPU matmul happens in bf16
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

  def test_dynamic_shape_multiple_batchs(self):
    torch_xla.manual_seed(100)
    device = torch_xla.device()
    # model setup
    in_dim = 16
    out_dum = 32
    batch = 8
    dummy_linear, dummy_linear_xla, input, input_xla = self._get_linear_and_input(
        in_dim, out_dum, batch, device)
    compiled_linear_xla = torch.compile(
        dummy_linear_xla, backend="openxla", dynamic=True)
    xm.wait_device_ops()
    met.clear_all()

    # first run with batch 8
    res_xla = compiled_linear_xla(input_xla)
    # torch.compile should be called once
    xm.wait_device_ops()
    self.assertEqual(met.counter_value('DynamoExtractCompiledGraph'), 1)
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # then run with batch 16
    met.clear_all()
    batch = 16
    input_xla = torch.randn(batch, in_dim).to(device)
    res_xla = compiled_linear_xla(input_xla)
    # torch.compile should not retrace but xla will recompile
    xm.wait_device_ops()
    self.assertNotIn('DynamoExtractCompiledGraph', met.counter_names())
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # then run with batch 32
    met.clear_all()
    batch = 32
    input_xla = torch.randn(batch, in_dim).to(device)
    res_xla = compiled_linear_xla(input_xla)
    # torch.compile should not retrace but xla will recompile
    xm.wait_device_ops()
    self.assertNotIn('DynamoExtractCompiledGraph', met.counter_names())
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # run with batch 8 again and make sure we don't recompile HLO
    met.clear_all()
    batch = 8
    input = torch.randn(batch, in_dim)
    input_xla = input.to(device)
    res_xla = compiled_linear_xla(input_xla)
    res = dummy_linear(input)
    torch.allclose(res, res_xla.cpu(), atol=1e-2, rtol=1e-4)
    # torch.compile should not retrace, xla also will not compile
    self.assertNotIn('DynamoExtractCompiledGraph', met.counter_names())
    self.assertNotIn('CompileTime', met.metric_names())
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

  def test_dynamic_shape_mix_with_non_dynamic(self):
    torch_xla.manual_seed(100)
    device = torch_xla.device()
    # model setup
    in_dim = 15
    out_dum = 31
    out_dum_2 = 33
    batch = 8
    _, dummy_linear_xla, _, input_xla = self._get_linear_and_input(
        in_dim, out_dum, batch, device)
    dynamic_compiled_linear_xla = torch.compile(
        dummy_linear_xla, backend="openxla", dynamic=True)
    _, dummy_linear_xla_2, _, input_xla_2 = self._get_linear_and_input(
        in_dim, out_dum_2, batch, device)
    static_compiled_linear_xla = torch.compile(
        dummy_linear_xla_2, backend="openxla")
    xm.wait_device_ops()
    met.clear_all()

    # first run the dynamic compiled model
    res_xla = dynamic_compiled_linear_xla(input_xla)
    # torch.compile should be called once
    xm.wait_device_ops()
    self.assertEqual(met.counter_value('DynamoExtractCompiledGraph'), 1)
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # and then run dynamic compiled model with differnt batch size
    met.clear_all()
    batch = 32
    input_xla = torch.randn(batch, in_dim).to(device)
    res_xla = dynamic_compiled_linear_xla(input_xla)
    # torch.compile should not retrace but xla will recompile
    xm.wait_device_ops()
    self.assertNotIn('DynamoExtractCompiledGraph', met.counter_names())
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # now run the static compiled model
    met.clear_all()
    res_xla = static_compiled_linear_xla(input_xla_2)
    # torch.compile should be called
    xm.wait_device_ops()
    self.assertEqual(met.counter_value('DynamoExtractCompiledGraph'), 1)
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

    # run static compiled model with different batch size, we expect the dynamo
    # to retrace the model.
    met.clear_all()
    batch = 12
    input_xla_2 = torch.randn(batch, in_dim).to(device)
    res_xla = static_compiled_linear_xla(input_xla_2)
    # torch.compile should be called
    xm.wait_device_ops()
    self.assertEqual(met.counter_value('DynamoExtractCompiledGraph'), 1)
    self.assertEqual(met.metric_data('CompileTime')[0], 1)
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

  def test_dynamic_shape_symint_as_return(self):
    device = torch_xla.device()
    config = DecoderOnlyConfig()
    config.num_hidden_layers = 2
    config.hidden_size = 512
    seq_len = 512
    decoder_model = DecoderOnlyModel(config).to(device)
    compiled_decoder_model = torch.compile(
        decoder_model, backend="openxla", dynamic=True)
    xm.wait_device_ops()
    met.clear_all()
    for batch_size in [1, 2, 3, 4, 5]:
      input = torch.zeros(batch_size, seq_len, dtype=torch.int64).to(device)
      res = compiled_decoder_model(input)
    # For some reason `batch_size == 1` is a special case where output does not
    # have additional ints. We will have on compile for batch size 1 and one compile
    # for other batch sizes.
    self.assertEqual(met.counter_value('DynamoExtractCompiledGraph'), 2)

  def test_dynamic_shape_no_retracing(self):
    device = torch_xla.device()
    # model setup
    _, dummy_linear_xla, _, input_xla = self._get_linear_and_input(
        8, 10, 20, device)
    compiled_linear_xla = torch.compile(
        dummy_linear_xla, backend="openxla", dynamic=True)
    xm.wait_device_ops()
    met.clear_all()

    # first run
    res_xla = compiled_linear_xla(input_xla)
    # Dynamo execution should not trigger `CachedCompile` counter. If we do it likely
    # means we retrace the same fx multiple times.
    self.assertNotIn('CachedCompile', met.counter_names())

  @unittest.skip(
      "Skip right now because with torch._dynamo.config.inline_inbuilt_nn_modules = True, dynamic compiles takes minutes for resnet18."
  )
  def test_dynamic_shape_resnet18(self):
    device = torch_xla.device()

    sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=10)
    loader = self._get_loader(device, sample_count, batch_size=4)
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

    loader_new_shape = self._get_loader(device, sample_count, batch_size=2)
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
