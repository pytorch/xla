import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


def f(hx):
  i = 0
  while i < 10:
    hx = hx + 1
    i = i + 1
  return hx


device = xm.xla_device()

hx = torch.ones(1).to(device)

print("hx", hx)

result = f(hx)
print(result)
