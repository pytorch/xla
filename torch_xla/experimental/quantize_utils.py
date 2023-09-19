import torch
from torch.fx import GraphModule
from torch.utils import _pytree as pytree
import torch_xla.core.xla_model as xm
from torch_xla.experimental.quantized import (
    xla_quantize_per_channel,
    xla_quantize_per_tensor,
    xla_dequantize_per_channel,
    xla_dequantize_per_tensor,
)
from torch_xla.stablehlo import XLAExportInterpreter
from typing import Any, Tuple


def pt2e_reference_model_to_stablehlo(m: GraphModule, args: Tuple[Any,
                                                                  ...]) -> str:
  exported_model = torch.export.export(m, args)
  exported_model(*args)

  #   print("check exported reference module")
  #   exported_model.graph_module.graph.print_tabular()

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

  replace_xla_qdq(exported_model.graph_module)

  # TODO: replace with exported_program_to_stablehlo
  # Trace with XLA
  device = xm.xla_device()
  args = pytree.tree_map_only(torch.Tensor, lambda x: x.to(device=device), args)
  param_and_buffer_keys = exported_model.graph_signature.parameters + exported_model.graph_signature.buffers
  state_dict = pytree.tree_map_only(torch.Tensor, lambda x: x.to(device=device),
                                    exported_model.state_dict)
  param_buffer_values = (state_dict[key] for key in param_and_buffer_keys)
  num_mutations = len(exported_model.graph_signature.buffers_to_mutate)

  with torch.no_grad():
    res = XLAExportInterpreter(exported_model.graph_module, device).run(
        *param_buffer_values, *args, enable_io_processing=False)
    res = res[num_mutations:]
    stablehlo = xm.get_stablehlo(res)
    return stablehlo
