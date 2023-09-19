import torch
import torch_xla
from torch_xla.experimental.quantized import xla_quantize_per_channel, xla_dequantize_per_channel
from torch_xla.core import xla_model as xm

device = xm.xla_device()
# input = torch.rand(5, 2).to(device)
# scale = torch.tensor([2.0, 3.4]).to(device)
# zero_point = torch.tensor([3.0, 5.0]).to(device)
# quant_min = -128
# quant_max = 127
# dtype = str(
#     torch.int8)  # Have to use str, need to fugure out c++ type for torch.dtype

# out = xla_quantize_per_tensor(
#     input, scale, zero_point, quant_min, quant_max, dtype, axis=1)
# out = xla_dequantize_per_tensor(
#     out, scale, zero_point, quant_min, quant_max, dtype, axis=1)
# out = out * 2

# hlo = torch_xla._XLAC._get_xla_tensors_hlo([out])
# print(hlo)

# stablehlo = xm.get_stablehlo([out])
# print(stablehlo)

x = torch.rand((2, 5)).to(device)
y = torch.rand((2, 5)).to(device)

add = x + y
q_add = xla_quantize_per_channel(add, torch.tensor([1.0, 2.0]),
                                 torch.tensor([1, 1]), 0, -128, 127,
                                 str(torch.int8))

dq_add = xla_dequantize_per_channel(q_add, torch.tensor([1.0, 2.0]),
                                    torch.tensor([1, 1]), 0, -128, 127,
                                    str(torch.int8))

hlo = torch_xla._XLAC._get_xla_tensors_hlo([dq_add])
print(hlo)

stablehlo = xm.get_stablehlo([dq_add])
print(stablehlo)
