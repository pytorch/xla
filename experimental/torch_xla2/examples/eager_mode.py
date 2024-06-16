import torch_xla2
from torch import nn
from torch.nn import functional as F
import torch

xla_env = torch_xla2.default_env()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

m = MyModel()
m = xla_env.to_xla(m)

# Execute this model using torch
inputs = (torch.randn(3, 3, 28, 28), )
inputs = xla_env.to_xla(inputs)

print(m(*inputs))
print('---=====')

from torch_xla2.interop import jax_jit

@jax_jit
def model_func(param, inputs):
  return torch.func.functional_call(m, param, inputs)

print(model_func(m.state_dict(), inputs))

print('---=====')
with xla_env:
  m2 = MyModel()
  inputs = (torch.randn(3, 3, 28, 28), )
  print(m2(*inputs))



