"""Tensor constructor overrides"""
import functools
from typing import Optional, Sequence

import jax
import torch
import jax.numpy as jnp
from torch_xla2 import tensor
from torch_xla2.ops.ops_registry import register_torch_function_op
from torch_xla2.ops import op_base


def register_function(torch_func, **kwargs):
  return functools.partial(register_torch_function_op, torch_func, **kwargs)


@register_function(torch.tensor)
@op_base.convert_dtype(use_default_dtype=False)  # Attempt to infer type from elements
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
@op_base.convert_dtype()
def _ones(*size: int, dtype=None, **kwargs):
  return jnp.ones(size, dtype)


@register_function(torch.zeros)
@op_base.convert_dtype()
def _zeros(*size: int, dtype=None, **kwargs):
  return jnp.zeros(size, dtype)


@register_function(torch.eye)
@op_base.convert_dtype()
def _eye(n: int, m: Optional[int] = None, *, dtype=None, **kwargs):
  return jnp.eye(n, m, dtype=dtype)


@register_function(torch.full)
@op_base.convert_dtype()
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
  # TODO: handle torch.Size
  return jnp.full(size, fill_value, dtype=dtype)


@register_function(torch.allclose)
def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
  return jnp.allclose(input, other, rtol, atol, equal_nan)

@register_function(torch.angle)
def _torch_angle(input):
  return jnp.angle(input)


@register_function(torch.argsort)
def _torch_argsort(input, dim=-1, descending=False, stable=False):
  expanded = False
  if input == 0:
    # for self of rank 0:
    # torch.any(x, 0), torch.any(x, -1) works;
    # torch.any(x, 1) throws out of bounds, so it's
    # behavior is the same as a jnp array of rank 1
    expanded = True
    input = jnp.expand_dims(input, 0)
  res = jnp.argsort(input, axis=dim, descending=descending,
                     stable=stable)
  if expanded:
    res = res.squeeze()
  return res


@register_function(torch.einsum)
def _einsum(equation, *operands):
  assert isinstance(equation, str), 'Only accept str equation'
  return jnp.einsum(equation, *operands)
