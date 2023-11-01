import torch
from torch.library import Library, impl
import torch_xla
from typing import List

xla_quantized_lib = Library("xla_quantized", "DEF")

xla_quantized_lib.define(
    "xla_quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")


@impl(xla_quantized_lib, "xla_quantize_per_tensor", "CompositeExplicitAutograd")
def xla_quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int,
                            quant_min: int, quant_max: int, dtype: torch.dtype):
  return xla_quantize_(input, torch.tensor([scale]),
                       torch.tensor([zero_point], dtype=torch.float), quant_min,
                       quant_max, dtype)


@impl(xla_quantized_lib, "xla_quantize_per_tensor", "Meta")
def xla_quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int,
                            quant_min: int, quant_max: int, dtype: torch.dtype):
  return torch.empty_like(input, dtype=dtype)


xla_quantized_lib.define(
    "xla_quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")


@impl(xla_quantized_lib, "xla_quantize_per_channel",
      "CompositeExplicitAutograd")
def xla_quantize_per_channel(input: torch.Tensor, scale: torch.Tensor,
                             zero_point: torch.Tensor, axis: int,
                             quant_min: int, quant_max: int,
                             dtype: torch.dtype):
  return xla_quantize_(input, scale, zero_point, quant_min, quant_max, dtype,
                       axis)


@impl(xla_quantized_lib, "xla_quantize_per_channel", "Meta")
def xla_quantize_per_channel(input: torch.Tensor, scale: torch.Tensor,
                             zero_point: torch.Tensor, axis: int,
                             quant_min: int, quant_max: int,
                             dtype: torch.dtype):
  return torch.empty_like(input, dtype=dtype)


xla_quantized_lib.define(
    "xla_dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")


@impl(xla_quantized_lib, "xla_dequantize_per_tensor",
      "CompositeExplicitAutograd")
def xla_dequantize_per_tensor(input: torch.Tensor, scale: float,
                              zero_point: int, quant_min: int, quant_max: int,
                              dtype: torch.dtype):
  return xla_dequantize_(input, torch.tensor([scale]),
                         torch.tensor([zero_point], dtype=torch.float),
                         quant_min, quant_max, dtype)


@impl(xla_quantized_lib, "xla_dequantize_per_tensor", "Meta")
def xla_dequantize_per_tensor(input: torch.Tensor, scale: float,
                              zero_point: int, quant_min: int, quant_max: int,
                              dtype: torch.dtype):
  return torch.empty_like(input, dtype=torch.float32)


xla_quantized_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")


@impl(xla_quantized_lib, "xla_dequantize_per_channel",
      "CompositeExplicitAutograd")
def xla_dequantize_per_channel(input: torch.Tensor, scale: torch.Tensor,
                               zero_point: torch.Tensor, axis: int,
                               quant_min: int, quant_max: int,
                               dtype: torch.dtype):
  return xla_dequantize_(input, scale, zero_point, quant_min, quant_max, dtype,
                         axis)


@impl(xla_quantized_lib, "xla_dequantize_per_channel", "Meta")
def xla_dequantize_per_channel(input: torch.Tensor, scale: torch.Tensor,
                               zero_point: torch.Tensor, axis: int,
                               quant_min: int, quant_max: int,
                               dtype: torch.dtype):
  return torch.empty_like(input, dtype=torch.float32)


def xla_quantize_(input: torch.Tensor,
                  scale: torch.Tensor,
                  zero_point: torch.Tensor,
                  quant_min: int,
                  quant_max: int,
                  dtype: torch.dtype,
                  axis: int = -1):
  if scale.device.type == 'xla':
    scale_list = scale.cpu().numpy().tolist()
  else:
    scale_list = scale.numpy().tolist()

  if zero_point.device.type == 'xla':
    zero_point_list = zero_point.cpu().numpy().tolist()
  else:
    zero_point_list = zero_point.numpy().tolist()

  result = torch_xla._XLAC._xla_quantize_tensor(input, scale_list,
                                                    zero_point_list, quant_min,
                                                    quant_max, str(dtype), axis)
  return result


def xla_dequantize_(input: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    quant_min: int,
                    quant_max: int,
                    dtype: torch.dtype,
                    axis: int = -1):
  if scale.device.type == 'xla':
    scale_list = scale.cpu().numpy().tolist()
  else:
    scale_list = scale.numpy().tolist()

  if zero_point.device.type == 'xla':
    zero_point_list = zero_point.cpu().numpy().tolist()
  else:
    zero_point_list = zero_point.numpy().tolist()

  result = torch_xla._XLAC._xla_dequantize_tensor(input, scale_list,
                                                      zero_point_list,
                                                      quant_min, quant_max,
                                                      str(dtype), axis)
  return result
