import torch
import torch_xla
from typing import List


def xla_quantize_per_tensor(input: torch.Tensor, scale: torch.Tensor,
                            zero_point: torch.Tensor, quant_min: int,
                            quant_max: int, dtype: torch.dtype):
  if scale.device.type == 'xla':
    scale_list = scale.cpu().numpy().tolist()
  else:
    scale_list = scale.numpy().tolist()

  if zero_point.device.type == 'xla':
    zero_point_list = zero_point.cpu().numpy().tolist()
  else:
    zero_point_list = zero_point.numpy().tolist()

  result = torch_xla._XLAC._xla_quantize_per_tensor(input, scale_list,
                                                    zero_point_list, quant_min,
                                                    quant_max, dtype)
  return result
