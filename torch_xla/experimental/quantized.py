import torch
from torch.library import Library, impl
import torch_xla
from typing import List

quantized_decomposed_lib = Library("quantized_decomposed", "IMPL")


@impl(quantized_decomposed_lib, "quantize_per_tensor", "XLA")
def xla_quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int,
                            quant_min: int, quant_max: int, dtype: torch.dtype):
  return _xla_quantize(input, torch.tensor([scale]),
                       torch.tensor([zero_point], dtype=torch.float), quant_min,
                       quant_max, dtype)


@impl(quantized_decomposed_lib, "quantize_per_channel", "XLA")
def xla_quantize_per_channel(input: torch.Tensor, scale: torch.Tensor,
                             zero_point: torch.Tensor, axis: int,
                             quant_min: int, quant_max: int,
                             dtype: torch.dtype):
  return _xla_quantize(input, scale, zero_point, quant_min, quant_max, dtype,
                       axis)


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "XLA")
def xla_dequantize_per_tensor(input: torch.Tensor, scale: float,
                              zero_point: int, quant_min: int, quant_max: int,
                              dtype: torch.dtype):
  return _xla_dequantize(input, torch.tensor([scale]),
                         torch.tensor([zero_point], dtype=torch.float),
                         quant_min, quant_max, dtype)


@impl(quantized_decomposed_lib, "dequantize_per_channel", "XLA")
def xla_dequantize_per_tensor(input: torch.Tensor, scale: torch.Tensor,
                              zero_point: torch.Tensor, axis: int,
                              quant_min: int, quant_max: int,
                              dtype: torch.dtype):
  return _xla_dequantize(input, scale, zero_point, quant_min, quant_max, dtype,
                         axis)


def _unpack_tensor_to_list(t: torch.Tensor):
  if t.device.type == 'xla':
    return t.cpu().numpy().tolist()
  else:
    return t.numpy().tolist()


def _xla_quantize(input: torch.Tensor,
                  scale: torch.Tensor,
                  zero_point: torch.Tensor,
                  quant_min: int,
                  quant_max: int,
                  dtype: torch.dtype,
                  axis: int = -1):
  # Scale and zero_point need to be unpacked(materialized before enter LTC),
  # because the quant param will be attached to tensor Shape in HLO/StableHLO.
  return torch_xla._XLAC._xla_quantize_tensor(
      input, _unpack_tensor_to_list(scale), _unpack_tensor_to_list(zero_point),
      quant_min, quant_max, str(dtype), axis)


def _xla_dequantize(input: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    quant_min: int,
                    quant_max: int,
                    dtype: torch.dtype,
                    axis: int = -1):
  # Scale and zero_point need to be unpacked(materialized before enter LTC),
  # because the quant param will be attached to tensor Shape in HLO/StableHLO.
  return torch_xla._XLAC._xla_dequantize_tensor(
      input, _unpack_tensor_to_list(scale), _unpack_tensor_to_list(zero_point),
      quant_min, quant_max, str(dtype), axis)
