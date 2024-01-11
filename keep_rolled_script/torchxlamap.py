import torch
from functorch.experimental import control_flow
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


from torch_xla import stablehlo
from typing import List


def f(i, x):
  return x + i

def map1(x):
  return control_flow.map(f, torch.ones(10), x)

device = xm.xla_device()

args = torch.ones(1, requires_grad=True).to(device)
result = map1(args)

print(result)
