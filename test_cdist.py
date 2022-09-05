import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

device = xm.xla_device()
a = torch.rand(2,4,device=device)
b = torch.rand(2,4,device=device)
c = torch.cdist(a, b, p=1)
xm.mark_step()
print(a, b, c)
# print(met.metrics_report())

# a = torch.tensor([4.0, 3.0], device=device)
# b = torch.tensor([2.0, 2.0], device=device)
# c = torch.floor_divide(a, b)
# xm.mark_step()
# print(c)
print(met.metrics_report())


