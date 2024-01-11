import unittest 
import torch
from functorch.experimental import control_flow
# from torch_xla import stablehlo

def f(i, x):
  return x + i


def map1(x):
  return control_flow.map(f, torch.ones(10, 10), x)


args = torch.ones(1, requires_grad=True)

result = map1(args)

print("args", args)
print("result", result)
