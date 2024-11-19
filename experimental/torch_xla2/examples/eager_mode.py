import torch_xla2
from torch import nn
from torch.nn import functional as F
import torch

xla_env = torch_xla2.enable_globally()


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
m = m.to('jax')

# Execute this model using torch
inputs = torch.randn(3, 3, 28, 28, device='jax')

print(m(inputs))
print('---=====')

m_compiled = torch_xla2.compile(m)

print(m_compiled(inputs))


print('---')

