import torch
import torch_xla
from torch_xla.experimental.quantized import xla_quantize_per_tensor
from torch_xla.core import xla_model as xm

device = xm.xla_device()
input = torch.rand(10).to(device)
scale = torch.tensor([2.0]).to(device)
zero_point = torch.tensor([3.0]).to(device)
quant_min = -128
quant_max = 127
dtype = str(
    torch.int8)  # Have to use str, need to fugure out c++ type for torch.dtype

out = xla_quantize_per_tensor(input, scale, zero_point, quant_min, quant_max,
                              dtype)

hlo = torch_xla._XLAC._get_xla_tensors_hlo([out])
print(hlo)

# Generated HLO with custom call to StableHLO
# ENTRY %IrToHlo.4 (p0.1: f32[10]) -> (f32[10]) {
#   %p0.1 = f32[10]{0} parameter(0)
#   %custom-call.2 = f32[10]{0} custom-call(f32[10]{0} %p0.1), custom_call_target="stablehlo.uniform_quantize", backend_config="container: torch.int8,scale: [2],zero_point: [3],quant_min: -128,quant_max: 127,dtype: torch.int8"
#   ROOT %tuple.3 = (f32[10]{0}) tuple(f32[10]{0} %custom-call.2)
# }