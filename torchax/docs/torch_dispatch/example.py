import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import _pytree as pytree


class Subclass(torch.Tensor):

  def __new__(cls, raw_data, requires_grad=False):
    return torch.Tensor._make_subclass(
        cls,
        raw_data,
        require_grad=requires_grad,
    )

  def __init__(self, raw_data=None, requires_grad=False):
    # Store any provided user raw_data
    self.raw_data = raw_data

  @classmethod
  #def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}
    print(f'func is {func}')

    def unpack(x):
      if isinstance(x, Subclass):
        return x.raw_data
      return x

    (args, kwargs) = pytree.tree_map(unpack, (args, kwargs))
    res = func(*args, **kwargs)
    return pytree.tree_map_only(torch.Tensor, Subclass, res)

  def __str__(self):
    return f'Subclass of shape {self.shape}'

  def add(self, a):
    print('HERE: add')
    return super().add(a)

  __repr__ = __str__


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


model = MyModel()

x = torch.randn(10, 28 * 28)
x2 = Subclass(x)
print(model(x2))

x2.add(2)
