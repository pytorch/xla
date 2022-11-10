import torch
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
num_element = 10
t1 = torch.zeros([num_element], device=dev)
t1[0] = 1
# print('xw32 t1=', t1)
t2 = torch.nonzero(t1)
# print('xw32 t2=', t2)
t3 = t2.view(num_element)
# print('xw32 t3=', t3)
xm.mark_step()
