import os
import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch._export import capture_pre_autograd_graph
import torchvision
from torch_xla import stablehlo
import torch_xla.core.xla_model as xm
from torch_xla.tf_saved_model_integration import save_torch_module_as_tf_saved_model
import unittest

os.environ['STABLEHLO_BYTECODE_FROM_PRETTYPRINT'] = '1'


class PT2EExportTest(unittest.TestCase):

  def test_per_tensor_qdq(self):
    device = xm.xla_device()
    x = torch.randn(2, 3, 4, 5).to(device)
    x = torch.ops.quantized_decomposed.quantize_per_tensor(
        x, 0.4, 2, -128, 127, torch.int8)
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, 0.4, 2, -128, 127, torch.int8)
    stablehlo_txt = xm.get_stablehlo([x])
    self.assertTrue("stablehlo.uniform_quantize" in stablehlo_txt)
    self.assertTrue("stablehlo.uniform_dequantize" in stablehlo_txt)

  @unittest.skip("Currently Failing")
  def test_per_channel_qdq(self):
    device = xm.xla_device()
    x = torch.randn(2, 3, 4, 5).to(device)
    scale = torch.tensor([3.2, 5.3, -0.1, 10])
    zero_point = torch.tensor([1, 2, -1, -2])
    x = torch.ops.quantized_decomposed.quantize_per_channel(
        x, scale, zero_point, 2, -128, 127, torch.int8)
    x = torch.ops.quantized_decomposed.dequantize_per_channel(
        x, scale, zero_point, 2, -128, 127, torch.int8)
    stablehlo_txt = xm.get_stablehlo([x])
    self.assertTrue("stablehlo.uniform_quantize" in stablehlo_txt)
    self.assertTrue("stablehlo.uniform_dequantize" in stablehlo_txt)

  def test_resnet18(self):
    # Step 1: export resnet18
    args = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet18().eval()
    m = capture_pre_autograd_graph(m, args)

    # Step 2: Insert observers or fake quantize modules
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config())
    m = prepare_pt2e(m, quantizer)

    # Step 3: Quantize the model
    m = convert_pt2e(m)

    # Trace with torch/xla and export stablehlo
    exported = torch.export.export(m, args)
    stablehlo_gm = stablehlo.exported_program_to_stablehlo(exported)
    stablehlo_txt = stablehlo_gm.get_stablehlo_text('forward')
    # print(stablehlo_txt)
    self.assertTrue("stablehlo.uniform_quantize" in stablehlo_txt)
    self.assertTrue("stablehlo.uniform_dequantize" in stablehlo_txt)
    # Save as tf.saved_model
    # save_torch_module_as_tf_saved_model(m, args, '/tmp/tf_saved_model/tmp1')


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
