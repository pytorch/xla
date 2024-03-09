import torch
import torch_xla.core.xla_model as xm
from torch_xla.experimental.quantized import matmul_4bit, pack_4bit
import torch_xla


device = xm.xla_device()
torch.manual_seed(0)
weight = torch.randint(0,5, (2,4)).to(torch.int8)
print(weight)
packed_weight = pack_4bit(weight, torch.int8)
print(packed_weight)

x = torch.randint(0,5, (3,2)).to(torch.bfloat16).to(device)
weight = weight.to(device)
packed_weight = packed_weight.to(device)

matmul_int8 = torch.matmul(x, weight)
matmul_int4 = matmul_4bit(x, packed_weight)
hlo = torch_xla._XLAC._get_xla_tensors_hlo([matmul_int4])
print(hlo)

print(matmul_int8)
print(matmul_int4)

torch.allclose(matmul_int8, matmul_int4)

