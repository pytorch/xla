from types import MethodType

import torch
import torch_xla.core.xla_model as xm
from torch_xla.utils.checkpoint import checkpoint


def checkpoint_module(module):
  """
  Wrap a `module`'s `forward` method with gradient checkpointing (also called
  activation checkpointing) via `torch_xla.utils.checkpoint.checkpoint`.

  Note that it doesn't support keyword arguments to `forward` at this moment.
  """
  assert isinstance(module, torch.nn.Module)
  module._forward_before_wrap_no_grad_ckpt = module.forward
  module.forward = MethodType(
      (lambda m, *args: checkpoint(m._forward_before_wrap_no_grad_ckpt, *args)),
      module)
  return module


def dummy_all_gather(value, dim=0):
  """A dummy op for debugging with the same output shape as all_gather"""
  repeat_num = [1] * value.dim()
  repeat_num[dim] = xm.xrt_world_size()
  return value.repeat(tuple(repeat_num))


def dummy_all_reduce(reduce_type, inputs, scale=1.0):
  """A dummy op for debugging with the same output shape as all_reduce"""
  if isinstance(inputs, torch.Tensor):
    return inputs * scale
  return [t.mul_(scale) for t in inputs]


def dummy_reduce_scatter(reduce_type, input, scale, scatter_dim, shard_count):
  """A dummy op for debugging with the same output shape as reduce_scatter"""
  assert shard_count == xm.xrt_world_size()
  full_size = input.size(scatter_dim)
  shard_size = full_size // xm.xrt_world_size()
  begin = shard_size * xm.get_ordinal()
  end = begin + shard_size
  slices = [None] * input.dim()
  slices[scatter_dim] = slice(begin, end)
  return input[tuple(slices)] * scale
