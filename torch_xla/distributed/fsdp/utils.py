from types import MethodType

import torch
import torch_xla.core.xla_model as xm
from torch_xla.utils.checkpoint import checkpoint


def checkpoint_module(module):
  """
  Wrap a `module`'s `forward` method with gradient checkpointing (also called
  activation checkpointing) via `torch_xla.utils.checkpoint.checkpoint`.
  """

  def _xla_checkpointed_forward_no_kwargs(m, num_args, num_kwargs,
                                          *packed_args):
    # unpack packed_args into args and kwargs
    assert num_args + num_kwargs * 2 == len(packed_args)
    args = packed_args[:num_args]
    kwargs = packed_args[num_args:]
    kwargs = dict(zip(kwargs[:num_kwargs], kwargs[num_kwargs:]))
    return m._xla_checkpointed_forward_original(*args, **kwargs)

  def _forward_with_checkpoint(m, *args, **kwargs):
    # pack args and kwargs together as `torch_xla.utils.checkpoint.checkpoint`
    # doesn't support keyword arguments
    packed_args = args + tuple(kwargs.keys()) + tuple(kwargs.values())
    input_requires_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in packed_args)
    if input_requires_grad:
      outputs = checkpoint(m._xla_checkpointed_forward_no_kwargs, len(args),
                           len(kwargs), *packed_args)
    else:
      # No input requires gradients so we won't checkpoint this forward pass.
      # Note that `m`` might have parameters that require gradients, but they
      # are beyond what `torch_xla.utils.checkpoint.checkpoint` can handle.
      outputs = m._xla_checkpointed_forward_original(*args, **kwargs)
    return outputs

  assert isinstance(module, torch.nn.Module)
  # replace `module`'s forward method with its checkpointed version
  module._xla_checkpointed_forward_original = module.forward
  module._xla_checkpointed_forward_no_kwargs = MethodType(
      _xla_checkpointed_forward_no_kwargs, module)
  module.forward = MethodType(_forward_with_checkpoint, module)
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
