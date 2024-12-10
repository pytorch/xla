import os
import tempfile
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla.core.xla_model as xm
import torchvision
from torch.export import export_for_training
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from torch_xla import stablehlo
from torch_xla.utils.stablehlo_test_utils import has_tf_package

try:
  from torch_xla.tf_saved_model_integration import \
      save_torch_module_as_tf_saved_model
except ImportError:
  print("tf is not installed. The tf.saved_model tests will be skipped.")

_TORCH_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]

_TORCH_DEQUANTIZE_OPS = [
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
]


def count_fx_graph_nodes(g: torch.fx.Graph, op_dict: Dict[str, List[Callable]]):
  cnt = {}
  for k in op_dict.keys():
    cnt[k] = 0
  for node in g.nodes:
    for k, ops in op_dict.items():
      if node.target in ops:
        cnt[k] += 1
  return cnt


def count_qdq_ops(g: torch.fx.Graph):
  op_dict = {
      "quantize": _TORCH_QUANTIZE_OPS,
      "dequantize": _TORCH_DEQUANTIZE_OPS,
  }
  return count_fx_graph_nodes(g, op_dict)


class PT2EExportTest(unittest.TestCase):

  def test_per_tensor_qdq(self):
    device = xm.xla_device()
    x = torch.randn(2, 3, 4, 5).to(device)
    x = torch.ops.quantized_decomposed.quantize_per_tensor(
        x, 0.4, 2, -128, 127, torch.int8)
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, 0.4, 2, -128, 127, torch.int8)
    stablehlo_txt = xm.get_stablehlo([x])
    self.assertEqual(
        stablehlo_txt.count(
            'tensor<2x3x4x5x!quant.uniform<i8:f32, 4.000000e-01:2>>'), 2)
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_quantize"), 1)
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_dequantize"), 1)

  def test_per_channel_qdq(self):
    device = xm.xla_device()
    x = torch.randn(2, 3, 4, 5).to(device)
    scale = torch.tensor([3.2, 5.3, 0.1, 10]).to(device)
    zero_point = torch.tensor([1, 2, -1, -2], dtype=torch.int64).to(device)
    x = torch.ops.quantized_decomposed.quantize_per_channel(
        x, scale, zero_point, 2, -128, 127, torch.int8)
    x = torch.ops.quantized_decomposed.dequantize_per_channel(
        x, scale, zero_point, 2, -128, 127, torch.int8)
    stablehlo_txt = xm.get_stablehlo([x])
    self.assertEqual(
        stablehlo_txt.count(
            'tensor<2x3x4x5x!quant.uniform<i8:f32:2, {3.200000e+00:1,5.300000e+00:2,1.000000e-01:-1,1.000000e+01:-2}>>'
        ), 2)
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_quantize"), 1)
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_dequantize"), 1)

  def test_resnet18(self):
    # Step 1: export resnet18
    args = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet18().eval()
    m = export_for_training(m, args).module()

    # Step 2: Insert observers or fake quantize modules
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config())
    m = prepare_pt2e(m, quantizer)

    # Step 3: Quantize the model
    m = convert_pt2e(m, fold_quantize=False)

    # Trace with torch/xla and export stablehlo
    exported = torch.export.export(m, args)
    stablehlo_gm = stablehlo.exported_program_to_stablehlo(exported)
    stablehlo_txt = stablehlo_gm.get_stablehlo_text()
    # fx_node_cnt = count_qdq_ops(exported.graph_module.graph)
    # Do not compare the number of qdq with the qdq in FX Graph.
    # In FX Graph, there will be 2 same dq ops in the backbone path
    # and the residule path.
    # The redundant dq ops will be removed by StableHLO
    # CanonicalizerPass/CSE Pass.
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_quantize"), 54)
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_dequantize"), 54)
    # Save as tf.saved_model
    if has_tf_package():
      tmp_path = tempfile.mkdtemp()
      save_torch_module_as_tf_saved_model(m, args, tmp_path)
      self.assertTrue(os.path.exists(os.path.join(tmp_path, 'saved_model.pb')))

  def test_resnet18_per_channel(self):
    # Step 1: export resnet18
    args = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet18().eval()
    m = export_for_training(m, args).module()

    # Step 2: Insert observers or fake quantize modules
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True))
    m = prepare_pt2e(m, quantizer)
    # Step 3: Run through example inputs, otherwise per-channel
    # quant may have scalar scale/zero_point
    m(*args)
    # Step 4: Quantize the model
    m = convert_pt2e(m, fold_quantize=False)

    # Trace with torch/xla and export stablehlo
    exported = torch.export.export(m, args)
    stablehlo_gm = stablehlo.exported_program_to_stablehlo(exported)
    stablehlo_txt = stablehlo_gm.get_stablehlo_text()
    # fx_node_cnt = count_qdq_ops(exported.graph_module.graph)
    # Do not compare the number of qdq with the qdq in FX Graph.
    # In FX Graph, there will be 2 same dq ops in the backbone path
    # and the residule path.
    # The redundant dq ops will be removed by StableHLO
    # CanonicalizerPass/CSE Pass.
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_quantize"), 54)
    self.assertEqual(stablehlo_txt.count("stablehlo.uniform_dequantize"), 54)
    # Save as tf.saved_model
    if has_tf_package():
      tmp_path = tempfile.mkdtemp()
      save_torch_module_as_tf_saved_model(m, args, tmp_path)
      self.assertTrue(os.path.exists(os.path.join(tmp_path, 'saved_model.pb')))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
