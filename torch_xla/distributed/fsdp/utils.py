from types import MethodType

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
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


def dummy_all_gather(value, dim=0, groups=None):
  """A dummy op for debugging with the same output shape as all_gather"""
  repeat_num = [1] * value.dim()
  repeat_num[dim] = xr.world_size()
  return value.repeat(tuple(repeat_num))


def dummy_all_reduce(reduce_type, inputs, scale=1.0, groups=None):
  """A dummy op for debugging with the same output shape as all_reduce"""
  if isinstance(inputs, torch.Tensor):
    return inputs * scale
  return [t.mul_(scale) for t in inputs]


class DummyReduceScatter:
  """A dummy op for debugging with the same output shape as reduce_scatter"""

  def __init__(self, shard_count):
    assert shard_count == xr.world_size()
    self.scale = 1.0

  def __call__(self, input, callback):
    full_size = input.size(0)
    shard_size = full_size // xr.world_size()
    begin = shard_size * xr.global_ordinal()
    end = begin + shard_size
    slices = [None] * input.dim()
    slices[0] = slice(begin, end)
    callback(input[tuple(slices)])

  def flush(self):
    pass


class BucketizedReduceScatter:
  """A reduce_scatter op that group input tensors before reduce-scattering them."""

  def __init__(self, bucket_size_mb, shard_count, groups, pin_layout) -> None:
    self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
    self.shard_count = shard_count
    self.groups = groups
    self.pin_layout = pin_layout
    self.scale = 1.0

    self.callbacks = []
    self.bucket = []
    self.bucket_watermark = 0

  def __call__(self, input, callback):
    input_byte_size = input.element_size() * input.numel()
    self.bucket.append(input)
    self.callbacks.append(callback)
    self.bucket_watermark += input_byte_size
    # If bucket_size_mb is default 0, flush for every tensor rather than coalesce
    if self.bucket_watermark > self.bucket_size_bytes:
      self.flush()

  def flush(self):
    if not self.bucket:
      return
    # TODO: debug coalesce error "" for GPU when pin_layout=True.
    # For now, workaround by using the non-coalesce version of reduce-scatter
    # when there's only 1 tensor input (bucket_size_mb=0).
    if len(self.bucket) == 1:
      result = xm.reduce_scatter(
          xm.REDUCE_SUM,
          self.bucket[0],
          scale=self.scale,
          scatter_dim=0,
          shard_count=self.shard_count,
          groups=self.groups,
          pin_layout=self.pin_layout)
      self.callbacks[0](result)
    else:
      results = xm.reduce_scatter(
          xm.REDUCE_SUM,
          self.bucket,
          scale=self.scale,
          scatter_dim=0,
          shard_count=self.shard_count,
          groups=self.groups,
          pin_layout=self.pin_layout)
      for cb, result in zip(self.callbacks, results):
        cb(result)

    self.bucket.clear()
    self.callbacks.clear()
    self.bucket_watermark = 0


class XLAPatchedLinear(torch.autograd.Function):
  """
  A patched version of `torch.nn.functional.linear` with explicitly-defined backward
  as a workaround to https://github.com/pytorch/xla/issues/3811.

  Modified from https://pytorch.org/docs/stable/notes/extending.html#example
  """

  @staticmethod
  def forward(ctx, input, weight, bias=None):
    # bias is an optional argument
    ctx.save_for_backward(input, weight, bias)
    with torch.no_grad():
      return torch._C._nn.linear(input, weight, bias)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    input_dim = input.dim()
    if input_dim > 2:
      input_flat = input.flatten(start_dim=0, end_dim=-2)
      grad_output_flat = grad_output.flatten(start_dim=0, end_dim=-2)
    else:
      input_flat = input
      grad_output_flat = grad_output

    if ctx.needs_input_grad[0]:
      grad_input_flat = grad_output_flat.mm(weight)
      if input_dim > 2:
        grad_input = grad_input_flat.view(*input.size())
      else:
        grad_input = grad_input_flat
    if ctx.needs_input_grad[1]:
      grad_weight = grad_output_flat.t().mm(input_flat)
    if bias is not None and ctx.needs_input_grad[2]:
      grad_bias = grad_output_flat.sum(0)

    return grad_input, grad_weight, grad_bias


def _xla_patched_nn_linear_forward(m, input):
  return XLAPatchedLinear.apply(input, m.weight, m.bias)


def apply_xla_patch_to_nn_linear(module,
                                 patched_function=_xla_patched_nn_linear_forward
                                ):
  """
  Recursively apply a patch to the forward pass of `nn.Linear` layers
  to enable using `XLAPatchedLinear.apply` as `torch.nn.functional.linear`,
  so that the backward pass will explicitly use the weight parameter of an
  `nn.Linear` layer to resolve https://github.com/pytorch/xla/issues/3811.

  Without this patch, an `nn.Linear` module in PyTorch/XLA holds and uses
  an intermediate result (rather than the weight parameter) in its backward
  computation, which may break the FSDP's full parameter freeing on it.
  """

  def _try_patching_forward_method(m, forward_method_name="forward"):
    # Check if the module's forward signature is same as in `nn.Linear`
    # (if it has already been modified through other means, we will skip the
    # patch to its forward method here).
    forward_method = getattr(m, forward_method_name, None)
    if forward_method is None:
      return
    if getattr(forward_method, "__func__", None) != torch.nn.Linear.forward:
      return

    patched_forward_method = MethodType(patched_function, m)
    m._nn_linear_forward_original = forward_method
    setattr(m, forward_method_name, patched_forward_method)

  for m in module.modules():  # includes self
    if isinstance(m, torch.nn.Linear):
      _try_patching_forward_method(m, "forward")
      # also handle the case of gradient checkpointing via `checkpoint_module`
      _try_patching_forward_method(m, "_xla_checkpointed_forward_original")

  return module
