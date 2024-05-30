import time

import torch
import numpy as np
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
from torch_xla import runtime as xr

server = xp.start_server(9012)

xr.use_spmd()
device = xm.xla_device()
# device = 'cpu'
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(42)

@xp.trace_me("Eager gmm")
def eager_gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor
) -> torch.Tensor:
  start = 0
  out = []
  for i, size in enumerate(group_sizes):
    result = lhs[start:start + size, :] @ rhs[i, :, :]
    out.append(result)
    start += group_sizes[i]
  return torch.cat(out)

@xp.trace_me("Eager tgmm")
def eager_tgmm(lhs: torch.Tensor, rhs: torch.Tensor,
              group_sizes: torch.Tensor) -> torch.Tensor:
  start = 0
  out = []
  for i, size in enumerate(group_sizes):
    result = lhs[:, start:start + size] @ rhs[start:start + size, :]
    out.append(result)
    start += group_sizes[i]
  return torch.stack(out)

@xp.trace_me("brute force gmm")
def brute_force_gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor
) -> torch.Tensor:
  """
  Just a simulation. Don't produce actual results as there is no masking.
  """
  out = 0
  for i, size in enumerate(group_sizes):
    result = lhs @ rhs[i, :, :]
    out += result
  return out

@xp.trace_me("brute force tgmm")
def brute_force_tgmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor
) -> torch.Tensor:
  """
  Just a simulation. Don't produce actual results as there is no masking.
  """
  out = []
  for i, size in enumerate(group_sizes):
    result = lhs @ (rhs + i)
    out.append(result)
  return torch.stack(out)

def time_execution(func, *args, backward=False, sharding_spec=None):
  start_time = time.time()
  lhs.grad = None
  rhs.grad = None

  o = func(*args)
  if backward:
    loss = nn.MSELoss()(o, torch.ones_like(o))
    loss.backward()
  xm.mark_step()
  xm.wait_device_ops()
  end_time = time.time()
  return end_time - start_time


def repeat_n(func, itr=10):
  sum = 0
  for _ in range(itr):
    sum += func()
  print(f"Execution time: {sum * 1000:.2f} ms")


from torch_xla.experimental.custom_kernel import gmm, _histogram, tgmm, gmm_backward

@xp.trace_me("gmm")
def gmm_kernel(*args):
  return gmm(*args)

@xp.trace_me("tgmm")
def tgmm_kernel(lhs, rhs, group_sizes, *args):
  lhs = lhs + 0
  rhs = rhs + 0
  lhs = xs.enable_manual_sharding(lhs, (None, 0)).global_tensor
  rhs = xs.enable_manual_sharding(rhs, (0, None)).global_tensor
  # group_sizes = xs.enable_manual_sharding(group_sizes, (None,)).global_tensor
  out = tgmm(lhs, rhs, group_sizes, *args)
  # compiler want this to be replicated instead of sharded. AllReduce instead of ReduceScatter.
  out = xs.disable_manual_sharding(out, (None, None, None), (group_sizes.shape[0], lhs.shape[0], rhs.shape[1])).global_tensor

  out = out + 0
  xs.mark_sharding(out, mesh, (None, None, 0))
  return out

def _group_sizes_strategy(m: int, num_groups: int) -> torch.Tensor:
  # Randomly sample the ends of the groups in the m-dimension. Let the fuzzer
  # sample with replacement so that it's possible to get zero-sized groups. Get
  # 'num_groups - 1' run ends. The final group will end at 'm'.
  ends_no_final = np.sort(
      np.array(
          [np.random.randint(low=0, high=m) for _ in range(num_groups - 1)],
          dtype=np.int32,
      ),)
  ends = np.concatenate([ends_no_final, np.array([m], dtype=np.int32)])

  # Calculate the run starts by shifting ends 1 to the right. The first run
  # starts at zero.
  starts = np.concatenate([np.zeros(1, dtype=np.int32), ends_no_final])
  return torch.from_numpy(ends - starts).to(torch.int32)


def routed_gmm(lhs, rhs):
  device = lhs.device
  m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]

  # Create TopK
  top1 = torch.randint(0, rhs.shape[0], (lhs.shape[0], 1)).to(device)
  top2 = torch.randint(0, rhs.shape[0], (lhs.shape[0], 1)).to(device)
  top = torch.cat([top1, top2], dim=1)

  # Enter manual sharding zone
  lhs = xs.enable_manual_sharding(lhs, (0, None)).global_tensor
  rhs = rhs + 0  # To create a new node such that rhs can keep its sharding.
  rhs = xs.enable_manual_sharding(rhs, (None, None, None)).global_tensor
  top = xs.enable_manual_sharding(top, (0, None)).global_tensor

  # We want to create one big batch of tokens that has all top-k choices in it.
  # Our tokens will thus be duplicated k-times in the batch. To do this we,
  # first flatten the expert choices list and argsort it. This gives us an array
  # of length B * K. We then create a tiled arange of size B * K and index
  # into the expert choices list. This will give us the set of indices we need
  # to gather from the xs to create this big batch.
  top_flat = top.flatten()
  lhs_order = top_flat.argsort()
  lhs_reverse_order = lhs_order.argsort()
  lhs_indices = torch.arange(lhs.shape[0], device=device).repeat_interleave(2)[lhs_order]  # Always replicated, so okay to skip manual sharding.
  lhs_sorted = lhs[lhs_indices]

  group_sizes = _histogram(top_flat.to(torch.int32), 0, rhs.shape[0] - 1)
  out = gmm_kernel(lhs_sorted, rhs, group_sizes)
  out = out[lhs_reverse_order].reshape(-1, 2, out.shape[-1]).sum(dim=1)

  # Exit manual sharding zone
  out = xs.disable_manual_sharding(out, (0, None), (m, n))
  return out


class RoutedGMMFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, lhs, rhs):
    device = lhs.device
    m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]

    # Create TopK
    top1 = torch.randint(0, rhs.shape[0], (lhs.shape[0], 1)).to(device)
    top2 = torch.randint(0, rhs.shape[0], (lhs.shape[0], 1)).to(device)
    top = torch.cat([top1, top2], dim=1)

    # To create a new node such that lhs/rhs can keep its sharding.
    lhs = lhs + 0
    rhs = rhs + 0

    # Saved for backward
    ctx.save_for_backward(lhs, rhs)

    # Enter manual sharding zone
    lhs = xs.enable_manual_sharding(lhs, (0, None)).global_tensor
    rhs = xs.enable_manual_sharding(rhs, (None, None, None)).global_tensor
    top = xs.enable_manual_sharding(top, (0, None)).global_tensor

    # We want to create one big batch of tokens that has all top-k choices in it.
    # Our tokens will thus be duplicated k-times in the batch. To do this we,
    # first flatten the expert choices list and argsort it. This gives us an array
    # of length B * K. We then create a tiled arange of size B * K and index
    # into the expert choices list. This will give us the set of indices we need
    # to gather from the xs to create this big batch.
    top_flat = top.flatten()
    lhs_order = top_flat.argsort()
    lhs_reverse_order = lhs_order.argsort()
    lhs_indices = torch.arange(lhs.shape[0], device=device).repeat_interleave(2)[lhs_order]  # Always replicated, so okay to skip manual sharding.
    lhs_sorted = lhs[lhs_indices]

    group_sizes = _histogram(top_flat.to(torch.int32), 0, rhs.shape[0] - 1)
    out = gmm_kernel(lhs_sorted, rhs, group_sizes)
    out = out[lhs_reverse_order].reshape(-1, 2, out.shape[-1]).sum(dim=1)

    # Exit manual sharding zone
    out = xs.disable_manual_sharding(out, (0, None), (m, n))

    # Saved for backward
    ctx.lhs_indices = lhs_indices
    ctx.lhs_reverse_order = lhs_reverse_order
    ctx.group_sizes = group_sizes
    return out

  @staticmethod
  def backward(ctx, grad_output):
    lhs_full, rhs_full = ctx.saved_tensors
    lhs_indices = ctx.lhs_indices
    lhs_reverse_order = ctx.lhs_reverse_order
    group_sizes = ctx.group_sizes

    # Enter manual sharding zone
    lhs = xs.enable_manual_sharding(lhs_full, (0, None)).global_tensor
    rhs = xs.enable_manual_sharding(rhs_full, (None, None, None)).global_tensor
    grad_output = xs.enable_manual_sharding(grad_output, (0, None)).global_tensor

    grad_sum = grad_output.unsqueeze(1).expand(-1, 2, -1)
    grad_reshape = grad_sum.reshape(-1, grad_sum.shape[-1])
    grad_index = grad_reshape[lhs_indices]
    grad_lhs_sorted, grad_rhs = gmm_backward(grad_index, lhs, rhs, group_sizes)
    grad_lhs_sorted = grad_lhs_sorted[lhs_reverse_order]
    grad_lhs = grad_lhs_sorted.reshape(-1, 2, grad_lhs_sorted.shape[-1]).sum(dim=1)


    # Exit manual sharding zone
    grad_lhs = xs.disable_manual_sharding(grad_lhs, (0, None), lhs_full.shape)
    grad_rhs = xs.disable_manual_sharding(grad_rhs, (None, None, None), rhs_full.shape)

    return grad_lhs, grad_rhs


n_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(range(n_devices), (n_devices, 1))
xs.set_global_mesh(mesh)
lhs = torch.randn(8192 * n_devices, 4096, requires_grad=True).to(device)  # 2 * 4096
rhs = torch.randn(8, 4096, 14336, requires_grad=True).to(device)
# lhs = torch.randn(4096, 16384 * n_devices).to(device)  # 2 * 2 * 4096
# rhs = torch.randn(16384 * n_devices, 14336).to(device)

xp.trace_detached('localhost:9012', '.', 30000)
# lhs = torch.randn(128, 128, requires_grad=True).to(device)
# rhs = torch.randn(8, 128, 128, requires_grad=True).to(device)
lhs.retain_grad()
rhs.retain_grad()

xs.mark_sharding(lhs, mesh, (0, None))
xs.mark_sharding(rhs, mesh, (None, 0, None))

# out = RoutedGMMFunction.apply(lhs, rhs)
# out = RoutedGMMFunction.forward(RoutedGMMFunction, lhs, rhs)
# out.sum().backward()
# print(out, lhs.grad, rhs.grad)

for i in range(5):
  # group_sizes = _group_sizes_strategy(lhs.shape[1], 8).to(torch.int32).to(device)
  # print(f"group_sizes: {group_sizes}")
  print(f"iteration: {i}")

  # Warm up
  repeat_n(lambda: time_execution(RoutedGMMFunction.apply, lhs, rhs, backward=True), itr=1)
  repeat_n(lambda: time_execution(RoutedGMMFunction.apply, lhs, rhs, backward=True))

# xm.mark_step()
# print(met.metrics_report())
