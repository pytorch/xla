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


# intermediate_size = 28672
# hidden_size = 8192
# token_size = 1024
intermediate_size = 16
hidden_size = 8
token_size = 8


class FFN(nn.Module):

  def __init__(self, world_size, hidden_dim, intermediate_dim, dtype=torch.bfloat16):
    super().__init__()
    self.fc1 = nn.Linear(hidden_dim, intermediate_dim // world_size, dtype=dtype, bias=False)
    self.fc2 = nn.Linear(intermediate_dim // world_size, hidden_dim, dtype=dtype, bias=False)
    self.relu = nn.ReLU()
    self.world_size = world_size
    self.groups = [[i for i in range(world_size)]]

  def forward(self, x):
    # x = xm.all_gather(x, dim=0, groups=self.groups, pin_layout=False)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    # x = xm.all_reduce(xm.REDUCE_SUM, x, scale=1.0, groups=self.groups, pin_layout=False)
    x = xm.all_reduce(xm.REDUCE_SUM, x)
    # x = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=self.world_size, groups=self.groups, pin_layout=False)
    return x


state_dict_path = "ffn_state_dict.pt"
input_path = "ffn_input.pt"
xla_out_path = "ffn_xla_out.pt"


def load_weights():
  full_weight_state_dict = torch.load(state_dict_path)
  input = torch.load(input_path)
  xla_out = torch.load(xla_out_path)
  return full_weight_state_dict, input, xla_out


def load_state_dict_shard(model, state_dict, index, world_size):
  fc1_full = state_dict['fc1.weight']
  fc2_full = state_dict['fc2.weight']
  
  per_chip_col_size = intermediate_size // world_size
  per_chip_row_size = intermediate_size // world_size
  print(f"per_chip_row_size: {per_chip_row_size}")

  col_parallel_start_idx = index * per_chip_col_size
  col_parallel_end_idx = col_parallel_start_idx + per_chip_col_size
  row_parallel_start_idx = index * per_chip_row_size
  row_parallel_end_idx = row_parallel_start_idx + per_chip_row_size
  model.fc1.weight.data = fc1_full[
      col_parallel_start_idx:col_parallel_end_idx, :]
  model.fc2.weight.data = fc2_full[:,
                                  row_parallel_start_idx:row_parallel_end_idx]
  return model


def load_input(input_full, index, world_size):
  total_seq_len = input_full.shape[0]
  local_seq_len = total_seq_len // world_size
  idx_start = index * local_seq_len
  idx_end = idx_start + local_seq_len
  return input_full[idx_start:idx_end]


def calc_torch_cpu(input, full_w1, full_w2):
  res = input @ full_w1.t()
  res = torch.nn.functional.relu(res)
  res = res @ full_w2.t()
  return res

def _mp_fn(index):
  device = xm.xla_device()
  world_size = xr.world_size()
  rank = xr.global_ordinal()
  # server = xp.start_server(9012)  # noqa: F841

  # Profile
  profile_dir = "profiles"
  print(f"Profiling (results will be saved to '{profile_dir}')...")
  # Enable tracing on server
  # xp.trace_detached("localhost:9012",
  #                   profile_dir,
  #                   delay_ms=0,
  #                   duration_ms=10000)
  # time.sleep(1.0)

  dist.init_process_group('xla', init_method='xla://')

  # with torch.device(device):
  model = FFN(world_size, hidden_size, intermediate_size)
  model = load_state_dict_shard(model, torch.load(state_dict_path), index, world_size)
  # print(f"fc1 index: {index} {model.fc1.weight}")
  # print(f"fc2 index: {index} {model.fc2.weight}")
  model = model.to('xla')
  
  # input = load_input(torch.load(input_path), index, world_size)
  input = torch.load(input_path)
  input = input.to('xla')
  # input = torch.randn((token_size // world_size, hidden_size), dtype=torch.bfloat16)
  # xm.mark_step()
  # compiled_model = torch.compile(model, backend="openxla", fullgraph=True, dynamic=False)

  with torch.no_grad():
    collective_matmul_output = model(input)
    xm.mark_step()
    xm.wait_device_ops()
  
  collective_matmul_output = collective_matmul_output.cpu()
  expected_xla_out = torch.load(xla_out_path)
  expected_xla_local_shard = load_input(expected_xla_out, index, world_size)
  print(f"expected_xla_out shape: {expected_xla_local_shard.shape}")
  print(f"collective_matmul_output shape: {collective_matmul_output.shape}")
  # print(expected_xla_local_shard - collective_matmul_output)
  # print(torch.load(input_path) / collective_matmul_output)
  # print(input.cpu() / collective_matmul_output)
  # print(expected_xla_out - collective_matmul_output)
  # print(f"shard {index} {collective_matmul_output[0][0]}")
  # print(f"expected_xla_out {expected_xla_out[0][0]}")

  # cpu_ref = calc_torch_cpu(torch.load(input_path), model.fc1.weight.data.cpu(), 
  #                          model.fc2.weight.data.cpu())
  sd = torch.load(state_dict_path)
  cpu_ref = calc_torch_cpu(torch.load(input_path), sd['fc1.weight'], 
                           sd['fc2.weight'])
  print(collective_matmul_output - cpu_ref)

if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
