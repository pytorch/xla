import torch_xla
import torch_xla.core.xla_model as xm
import torch
import torch._export
import unittest
from torch import nn

from typing import Tuple

Tensor = torch.Tensor

class ElementwiseAdd(nn.Module):

  def __init__(self) -> None:
    super().__init__()

  def forward(self, x: Tensor, y: Tensor) -> Tensor:
    return x + y

  def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
    return (torch.randn(1, 3), torch.randn(1, 3))

model = ElementwiseAdd()
inputs = model.get_random_inputs()

device = xm.xla_device()
model.eval()

print(list(inputs))
model = model.to(device)
inputs = tuple(i.to(device) for i in inputs if hasattr(i, 'to'))
output = model(*inputs)

bytecode = xm.get_stablehlo_bytecode([output])
res = torch_xla._XLAC._run_stablehlo(bytecode, list(inputs))

print(res)
