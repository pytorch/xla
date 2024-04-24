"""Tensor constructor overrides"""
import functools
import logging
from typing import Callable, Optional, ParamSpec, Sequence

import jax
import torch
import jax.numpy as jnp
from torch_xla2 import tensor

registry = {}

P = ParamSpec('P')


def register_function(torch_func: Callable[P, torch.Tensor]):
  """Registers a function as the JAX implementation of a torch function."""

  def decorator(jax_impl: Callable[P, jax.Array]):
    registry[torch_func] = jax_impl
    return jax_impl

  return decorator


def convert_dtype(use_default_dtype: bool = True):
  """Converts `dtype` kwarg of function from torch to JAX.

  Args:
    use_default_dtype: Whether to use torch default dtype if none is provided.

  Returns:
    A decorator that wraps a JAX implementation of a torch function.
  """

  def decorator(func: Callable[P, torch.Tensor]):

    @functools.wraps(func)
    def wrapper(*args: P.args,
                dtype: Optional[torch.dtype] = None,
                **kwargs: P.kwargs):
      if not dtype and use_default_dtype:
        dtype = torch.get_default_dtype()
      jax_dtype = tensor.t2j_dtype(dtype)

      return func(*args, dtype=jax_dtype, **kwargs)

    return wrapper

  return decorator


@register_function(torch.tensor)
@convert_dtype(use_default_dtype=False)  # Attempt to infer type from elements
def _tensor(data, *, dtype=None, **kwargs):
  python_types_to_torch_types = {
      bool: jnp.bool,
      int: jnp.int64,
      float: jnp.float32,
      complex: jnp.complex64,
  }
  if not dtype:
    leaves = jax.tree_util.tree_leaves(data)
    if len(leaves) > 0:
      dtype = python_types_to_torch_types.get(type(leaves[0]))

  return jnp.array(
      data, dtype=dtype or tensor.t2j_dtype(torch.get_default_dtype()))


@register_function(torch.ones)
@convert_dtype()
def _ones(*size: int, dtype=None, **kwargs):
  return jnp.ones(size, dtype)


@register_function(torch.zeros)
@convert_dtype()
def _zeros(*size: int, dtype=None, **kwargs):
  return jnp.zeros(size, dtype)


@register_function(torch.eye)
@convert_dtype()
def _eye(n: int, m: Optional[int] = None, *, dtype=None, **kwargs):
  return jnp.eye(n, m, dtype=dtype)


@register_function(torch.full)
@convert_dtype()
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
  # TODO: handle torch.Size
  return jnp.full(size, fill_value, dtype=dtype)


class XLAFunctionMode(torch.overrides.TorchFunctionMode):
  """Context manager that dispatches torch function calls to JAX."""

  def __torch_function__(self,
                         func,
                         types,
                         args=(),
                         kwargs=None) -> torch.Tensor:
    jax_func = registry.get(func)
    if not jax_func:
      return func(*args, **(kwargs or {}))

    # TODO: unwrap args here or in implementations?
    return tensor.wrap(jax_func(*args, **(kwargs or {})))
