"""Tensor constructor overrides"""
import logging
import warnings

import torch
import jax.numpy as jnp
from torch_xla2 import tensor

# TODO: registry
# TODO: correct types
def ones(*size: int):
  return jnp.ones(size)


fns = {
  torch.tensor: jnp.array,
  torch.ones: ones,
  # torch.zeros: jnp.zeros,
  # torch.arange: jnp.arange,
  # torch.linspace: jnp.linspace,
  # torch.logspace: jnp.logspace,
  # torch.empty: jnp.empty,
  # torch.eye: jnp.eye,
  # torch.full: jnp.full,
}

class XLAFunctionMode(torch.overrides.TorchFunctionMode):
  def __torch_function__(self, func, types, args=(), kwargs=None) -> torch.Tensor:
    jax_func = fns.get(func)
    if not jax_func:
      logging.warn(f'Falling back to default implementation of {func.__name__}')
      return func(*args, **(kwargs or {}))

    if kwargs:
      warnings.warn(f'kwargs not implemented for {kwargs}')

    return tensor.wrap(jax_func(*tensor.unwrap(args)))
