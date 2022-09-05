import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

device = xm.xla_device()
a = torch.rand(200,4,device=device)
b = torch.rand(200,4,device=device)
c = torch.cdist(a, b, p=1)
xm.mark_step()
print(met.metrics_report())
