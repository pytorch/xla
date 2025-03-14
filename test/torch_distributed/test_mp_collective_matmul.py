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


intermediate_size = 28672
hidden_size = 8192
token_size = 1024
state_dict_path = "ffn_state_dict.pt"
input_path = "ffn_input.pt"
xla_out_path = "ffn_xla_out.pt"

class FFN(nn.Module):

  def __init__(self, world_size, hidden_dim, intermediate_dim, dtype=torch.bfloat16):
    super().__init__()
    self.fc1 = nn.Linear(hidden_dim, intermediate_dim // world_size, dtype=dtype, bias=False)
    self.fc2 = nn.Linear(intermediate_dim // world_size, hidden_dim, dtype=dtype, bias=False)
    self.relu = nn.ReLU()
    self.world_size = world_size
    self.groups = [[i for i in range(world_size)]]
    self.groups = [[6, 4, 2, 0, 1, 3, 5, 7]]

  def forward(self, x):
    x = xm.all_gather(x, dim=0, groups=self.groups, pin_layout=False)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    x = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=self.world_size, groups=self.groups, pin_layout=False)
    return x


def load_state_dict_shard(model, state_dict, index, world_size):
  fc1_full = state_dict['fc1.weight']
  fc2_full = state_dict['fc2.weight']

  per_chip_col_size = intermediate_size // world_size
  per_chip_row_size = intermediate_size // world_size

  col_parallel_start_idx = index * per_chip_col_size
  col_parallel_end_idx = col_parallel_start_idx + per_chip_col_size
  row_parallel_start_idx = index * per_chip_row_size
  row_parallel_end_idx = row_parallel_start_idx + per_chip_row_size
  model.fc1.weight.data = fc1_full[
      col_parallel_start_idx:col_parallel_end_idx, :]
  model.fc2.weight.data = fc2_full[:,
                                  row_parallel_start_idx:row_parallel_end_idx]
  return model


def master_print(*msg):
  index = xr.global_ordinal()
  if index == 0:
    print(*msg)


def load_input(input_full, index, world_size):
  total_seq_len = input_full.shape[0]
  local_seq_len = total_seq_len // world_size
  idx_start = index * local_seq_len
  idx_end = idx_start + local_seq_len
  return input_full[idx_start:idx_end]


def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  index = xr.global_ordinal()

  dist.init_process_group('xla', init_method='xla://')

  model = FFN(world_size, hidden_size, intermediate_size)
  model = load_state_dict_shard(model, torch.load(state_dict_path), index, world_size)
  model = model.to(device)
  input = torch.load(input_path)
  input = load_input(input, index, world_size)
  input = input.to(device)
  xm.mark_step()

  with torch.no_grad():
    for _ in range(16):
      collective_matmul_output = model(input)
      xm.mark_step()
      xm.wait_device_ops()

  collective_matmul_output = collective_matmul_output.cpu()
  expected_xla_out = torch.load(xla_out_path)
  expected_xla_local_shard = load_input(expected_xla_out, index, world_size)
  master_print(f"expected_xla_out shape: {expected_xla_local_shard.shape}")
  master_print(f"collective_matmul_output shape: {collective_matmul_output.shape}")
  master_print(torch.allclose(collective_matmul_output, expected_xla_local_shard, rtol=2e-2, atol=2e-2))


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
