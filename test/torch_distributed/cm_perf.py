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


intermediate_size = 28672
hidden_size = 8192
token_size = 1024

class FFN(nn.Module):

  def __init__(self, world_size, hidden_dim, intermediate_dim, dtype=torch.bfloat16):
    super().__init__()
    self.fc1 = nn.Linear(hidden_dim, intermediate_dim // world_size, dtype=dtype, bias=False)
    self.fc2 = nn.Linear(intermediate_dim // world_size, hidden_dim, dtype=dtype, bias=False)
    self.relu = nn.ReLU()
    self.world_size = world_size
    self.groups = [[i for i in range(world_size)]]

  def forward(self, x):
    x = xm.all_gather(x, dim=0, groups=self.groups, pin_layout=False)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    x = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=self.world_size, groups=self.groups, pin_layout=False)
    return x
  
class Model(nn.Module):
  def __init__(self, world_size, hidden_dim, intermediate_dim, dtype=torch.bfloat16, layer_num=16):
    super().__init__()
    self.layers = []
    for _ in range(layer_num):
      self.layers.append(FFN(world_size, hidden_dim, intermediate_dim))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


def master_print(*msg):
  index = xr.global_ordinal()
  if index == 0:
    print(*msg)


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  index = xr.global_ordinal()
  server = xp.start_server(9012)  # noqa: F841

  # Profile
  profile_dir = "profiles"
  master_print(f"Profiling (results will be saved to '{profile_dir}')...")
  # Enable tracing on server
  xp.start_trace(profile_dir)

  dist.init_process_group('xla', init_method='xla://')

  with torch.device(device=device):
    model = Model(world_size, hidden_size, intermediate_size)
    input = torch.ones((token_size // world_size, hidden_size), dtype=torch.bfloat16)
  xm.mark_step()

  with torch.no_grad():
    for _ in range(16):
      collective_matmul_output = model(input)
      xm.mark_step()
      xm.wait_device_ops()

  xp.stop_trace()


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
