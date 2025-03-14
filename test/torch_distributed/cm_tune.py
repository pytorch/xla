import os
import sys
import time
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch.distributed as dist
from torch import nn
import torch_xla.debug.profiler as xp
import itertools


intermediate_size = 28672
hidden_size = 8192
token_size = 4096

class FFN(nn.Module):

  def __init__(self, world_size, hidden_dim, intermediate_dim, dtype=torch.bfloat16):
    super().__init__()
    self.fc1 = nn.Linear(hidden_dim, intermediate_dim // world_size, dtype=dtype, bias=False)
    self.fc2 = nn.Linear(intermediate_dim // world_size, hidden_dim, dtype=dtype, bias=False)
    self.relu = nn.ReLU()
    self.world_size = world_size

  def forward(self, x, groups):
    x = xm.all_gather(x, dim=0, groups=groups, pin_layout=False)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    x = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=self.world_size, groups=groups, pin_layout=False)
    return x
  
class Model(nn.Module):
  def __init__(self, world_size, hidden_dim, intermediate_dim, dtype=torch.bfloat16, layer_num=16):
    super().__init__()
    self.layers = []
    for _ in range(layer_num):
      self.layers.append(FFN(world_size, hidden_dim, intermediate_dim))

  def forward(self, x, groups):
    for layer in self.layers:
      x = layer(x, groups)
    return x


def master_print(*msg):
  index = xr.global_ordinal()
  if index == 0:
    print(*msg)


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  index = xr.global_ordinal()

  dist.init_process_group('xla', init_method='xla://')

  with torch.device(device=device):
    model = Model(world_size, hidden_size, intermediate_size)
    input = torch.ones((token_size // world_size, hidden_size), dtype=torch.bfloat16)
  xm.mark_step()


  def permutations_0_to_6():
    """Generates all permutations of the numbers 0 through 6."""
    numbers = list(range(7))  # Creates a list [0, 1, 2, 3, 4, 5, 6]
    for permutation in itertools.permutations(numbers):
      yield permutation

  perm2time = {}
  for i, perm in enumerate(permutations_0_to_6()):
    groups = [list(perm) + [7]]
    print("========== groups: ", i, groups)

    with torch.no_grad():
      for _ in range(16):
        collective_matmul_output = model(input, groups)
        xm.mark_step()
    xm.wait_device_ops()

    start_time = time.perf_counter()
    with torch.no_grad():
      for _ in range(16):
        collective_matmul_output = model(input, groups)
        xm.mark_step()
    xm.wait_device_ops()
    end_time = time.perf_counter()

    perm2time[tuple(groups[0])] = (end_time - start_time) * 1000
  
  kv = list(perm2time.items())
  kv.sort(key=lambda pair: pair[1])
  with open("perf.txt", 'w') as f:
    for pair in kv:
      f.write(f"{pair[0]}, {pair[1]}\n")



if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
