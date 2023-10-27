import os
import torch
from torch.fx import GraphModule
from torch_xla.experimental.quantized import (
    xla_quantize_per_channel,
    xla_quantize_per_tensor,
    xla_dequantize_per_channel,
    xla_dequantize_per_tensor,
)
from torch_xla import stablehlo
from torch_xla.tf_saved_model_integration import save_stablehlo_graph_as_tf
from typing import Any, Tuple

def replace_xla_qdq(g: torch.fx.GraphModule):
    _QUANTIZE_PER_TENSOR_OPS = [
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    ]
    _QUANTIZE_PER_CHANNEL_OPS = [
        torch.ops.quantized_decomposed.quantize_per_channel.default,
    ]
    _DEQUANTIZE_PER_TENSOR_OPS = [
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
    ]
    _DEQUANTIZE_PER_CHANNEL_OPS = [
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ]
    for n in g.graph.nodes:
      if n.op == "call_function":
        if n.target in _QUANTIZE_PER_TENSOR_OPS:
          n.target = xla_quantize_per_tensor
        elif n.target in _QUANTIZE_PER_CHANNEL_OPS:
          n.target = xla_quantize_per_channel
        elif n.target in _DEQUANTIZE_PER_TENSOR_OPS:
          n.target = xla_dequantize_per_tensor
        elif n.target in _DEQUANTIZE_PER_CHANNEL_OPS:
          n.target = xla_dequantize_per_channel

def pt2e_reference_model_to_stablehlo(m: GraphModule, args: Tuple[Any,
                                                                  ...], emit_bytecode=False) -> str:
  exported = torch.export.export(m, args)
  replace_xla_qdq(exported.graph_module)
  options = stablehlo.StableHLOExportOptions(override_tracing_arguments=args)
  stablehlo_model = stablehlo.exported_program_to_stablehlo(exported, options)

  if emit_bytecode:
    return stablehlo_model.get_stablehlo_bytecode()
  else:
    return stablehlo_model.get_stablehlo_text()

def pt2e_reference_model_to_tf_saved_model(m: GraphModule, args: Tuple[Any,
                                                                  ...],
                                          saved_model_dir: os.PathLike) -> str:
  exported = torch.export.export(m, args)
  replace_xla_qdq(exported.graph_module)
  options = stablehlo.StableHLOExportOptions(override_tracing_arguments=args)
  stablehlo_model = stablehlo.exported_program_to_stablehlo(exported, options)
  save_stablehlo_graph_as_tf(stablehlo_model, saved_model_dir)
