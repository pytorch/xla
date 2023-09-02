import torch
import torch_xla
from torch_xla.core import xla_model as xm


device = xm.xla_device()

from torchvision.models import resnet18
model = resnet18()
model.to(device)
input = torch.rand(1, 3, 224, 224).to(device)
torch_xla._XLAC._xla_set_tag(input, 'tag_of_input')
torch_xla._XLAC._xla_mark_dynamic(input, 0)
result = model(input)
print(xm.get_stablehlo([result]))

# ## multiply (same axis dynamic)
# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# ## multiply (all axes dynamic)
# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# torch_xla._XLAC._xla_mark_dynamic(a, 1)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# ## multiply (possible to infer static shapes)
# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# # ## multiply (implicit broadcast)
# a = torch.randn((10,1)).to(device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.randn((5)).to(device=device)
# torch_xla._XLAC._xla_set_tag(b, 'tag_of_b')
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a + b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# torch_xla._XLAC._xla_mark_dynamic(a, 1)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a + b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))


# a = torch.randn((5,10)).to(device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.randn((10,5)).to(device=device)
# torch_xla._XLAC._xla_set_tag(b, 'tag_of_b')
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# c = torch.randn((5,5)).to(device=device)
# torch_xla._XLAC._xla_set_tag(c, 'tag_of_c')
# torch_xla._XLAC._xla_mark_dynamic(c, 0)
# d = a @ b * c
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([d])
# print(hlo_content)
# print(xm.get_stablehlo([d]))
