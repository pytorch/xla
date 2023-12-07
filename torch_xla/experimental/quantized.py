import numpy as np
import torch
import torch_xla
from torch.library import Library, impl

quantized_decomposed_lib = Library("quantized_decomposed", "IMPL")


@impl(quantized_decomposed_lib, "quantize_per_tensor", "XLA")
def xla_quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int,
                            quant_min: int, quant_max: int, dtype: torch.dtype):
  return _xla_quantize(input, torch.tensor([scale]),
                       torch.tensor([zero_point], dtype=dtype), quant_min,
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
                         torch.tensor([zero_point], dtype=dtype), quant_min,
                         quant_max, dtype)


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


def _check_scale_zp(input, scale, zero_point, axis, dtype):
  # The followings are checked:
  # 1. scale, zp are 1D tensor.
  # 2. Lenghth of scale, zp matched the (de)quant dim.
  # 3. zp dtype is the same as the quantized integer type.
  assert len(scale.shape) == 1 and len(zero_point.shape) == 1
  assert zero_point.dtype == dtype
  if axis == -1:
    assert scale.numel() == 1 and zero_point.numel() == 1
  else:
    assert axis >= 0 and axis < len(input.shape)
    qdq_dim_size = input.shape[axis]
    assert qdq_dim_size == scale.numel() and qdq_dim_size == zero_point.numel()


def _xla_quantize(input: torch.Tensor,
                  scale: torch.Tensor,
                  zero_point: torch.Tensor,
                  quant_min: int,
                  quant_max: int,
                  dtype: torch.dtype,
                  axis: int = -1):
  _check_scale_zp(input, scale, zero_point, axis, dtype)
  # Scale and zero_point need to be unpacked(materialized before enter LTC),
  # because the quant param will be attached to tensor Shape in HLO/StableHLO.
  scale_np = _unpack_tensor_to_list(scale)
  zp_np = _unpack_tensor_to_list(zero_point)
  # All scaler values needs to be greater than 0. (StableHLO qdq op constraint)
  assert np.all(np.greater(scale_np, 0))
  return torch_xla._XLAC._xla_quantize_tensor(input, scale_np, zp_np, quant_min,
                                              quant_max, str(dtype), axis)


def _xla_dequantize(input: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    quant_min: int,
                    quant_max: int,
                    dtype: torch.dtype,
                    axis: int = -1):
  _check_scale_zp(input, scale, zero_point, axis, dtype)
  # Scale and zero_point need to be unpacked(materialized before enter LTC),
  # because the quant param will be attached to tensor Shape in HLO/StableHLO.
  scale_np = _unpack_tensor_to_list(scale)
  zp_np = _unpack_tensor_to_list(zero_point)
  # All scaler values needs to be greater than 0. (StableHLO qdq op constraint)
  assert np.all(np.greater(scale_np, 0))
  return torch_xla._XLAC._xla_dequantize_tensor(input, scale_np,
                                                zp_np, quant_min, quant_max,
                                                str(dtype), axis)
