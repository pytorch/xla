import time

import torch
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met
device = xm.xla_device()

server = xp.start_server(9012)


@xp.trace_me("Eager Attention")
def attention(q, k, v):
  attn_weight = q @ k.transpose(-2, -1)
  attn_weight = nn.functional.softmax(attn_weight, dim=-1)
  attn_output = attn_weight @ v
  return attn_output


def time_execution(q, k, v, func):
  o = func(q, k, v)
  loss = o.sum()
  loss.backward()
  xm.mark_step()
  start_time = time.time()
  xm.wait_device_ops()
  end_time = time.time()
  return end_time - start_time


def repeat_n(func, itr=10):
  sum = 0
  for _ in range(itr):
    sum += func()
  print(f"Execution time: {sum * 1000:.2f} ms")


from torch_xla.experimental.custom_kernel import flash_attention


def shape_fn(q, k, v):
  # return [(q.shape, q.dtype), ((1, 1, 128, 128),  q.dtype), ((1, 1, 128, 128),  q.dtype), ((1, 1, 128, 128),  q.dtype)]
  return (q.shape, q.dtype)


@xp.trace_me("Flash Attention")
def flash_attention_kernel(q, k, v):
  return flash_attention(q, k, v)


xp.trace_detached('localhost:9012', '.', 120000)

for seq_len in [1024, 2048, 4096, 8192, 16384]:
  print(f"seq_len: {seq_len}")
  # Simulates 70B Llama 2
  # bs, num_heads, seq_len, head_dim
  shape = (2, 64, seq_len, 128)
  # shape = (3, 2, 128, 4)
  q = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).to(device)
  k = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).to(device)
  v = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True).to(device)
  q.retain_grad()
  k.retain_grad()
  v.retain_grad()

  # repeat_n(lambda: time_execution(q, k, v, attention), itr=5)
  if seq_len < 8192:
    repeat_n(lambda: time_execution(q, k, v, attention))

  # repeat_n(lambda: time_execution(q, k, v, flash_attention_kernel), itr=5)
  repeat_n(lambda: time_execution(q, k, v, flash_attention_kernel))

# print(met.metrics_report())
# o = flash_attention_kernel(q, k, v)
# hlo = torch_xla._XLAC._get_xla_tensors_hlo([o])
# print(hlo)
