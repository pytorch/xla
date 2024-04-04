"""Tensor constructor overrides"""
import functools
import logging
from typing import Callable, Optional, ParamSpec, Sequence
import warnings

import jax
import torch
import jax.numpy as jnp
from torch_xla2 import tensor

registry = {}

P = ParamSpec('P')

def register_function(torch_func: Callable[P, torch.Tensor]):
  def decorator(jax_impl: Callable[P, jax.Array]):
    @functools.wraps(torch_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
      return jax_impl(*args, **kwargs)

    registry[torch_func] = jax_impl

    return wrapper
  return decorator


@register_function(torch.tensor)
def _tensor(data, *args, **kwargs):
  return jnp.array(data)

@register_function(torch.ones)
def _ones(*size: int, **kwargs):
  return jnp.ones(size)

@register_function(torch.zeros)
def _zeros(*size: int, **kwargs):
  return jnp.zeros(size)

@register_function(torch.eye)
def _eye(n: int, m: Optional[int] = None, **kwargs):
  return jnp.eye(n, m)

@register_function(torch.full)
def _full(size: Sequence[int], fill_value, **kwargs):
  # TODO: handle torch.Size
  return jnp.full(size, fill_value)

class XLAFunctionMode(torch.overrides.TorchFunctionMode):
  def __torch_function__(self, func, types, args=(), kwargs=None) -> torch.Tensor:
    jax_func = registry.get(func)
    if not jax_func:
      logging.warn(f'Falling back to default implementation of {func.__name__}')
      return func(*args, **(kwargs or {}))

    if kwargs:
      warnings.warn(f'kwargs not implemented for {kwargs}')

    # TODO: unwrap args here or in implementations?
    return tensor.wrap(jax_func(*args))
