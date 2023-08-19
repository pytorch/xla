import torch
import torch_xla
from torch_xla.core import xla_model as xm
from torch_xla.experimental import stablehlo_saved_model as sm


device = xm.xla_device()

a = torch.tensor([[1,2],[2,4]], device=device)
torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
torch_xla._XLAC._xla_mark_dynamic(a, 0)
b = torch.tensor([[1,2],[2,4]], device=device)
torch_xla._XLAC._xla_mark_dynamic(b, 0)
torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
c = a + b
print(xm.get_stablehlo([c]))
