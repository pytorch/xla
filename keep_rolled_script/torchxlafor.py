import torch
from functorch.experimental import control_flow
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


def f(x):
    for i in range(10):
        x = x + 1
    return x


# def map1(x):
#   return control_flow.map(f, torch.ones(10, 10), x)

device = xm.xla_device()

x = torch.ones(1).to(device)

result = f(x)
print(result)
