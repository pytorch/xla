import torch, torch_xla
import torch_xla.core.xla_model as xm
import torch.utils.dlpack

b_xla = torch.arange(10, device=xm.xla_device())
# b_cuda = b_xla.cuda()
t2 = torch.from_dlpack(b_xla)
