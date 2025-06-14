import functools
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchax.ops import mappings
from torchax.view import View
from torchax import types
import sys

from typing import Callable, Optional, ParamSpec, Concatenate


class InplaceOp:

  def __init__(self,
               functional_op,
               replace=False,
               position_to_mutate=0,
               is_jax_func=False):
    self.functional = functional_op
    self.replace = replace
    self.position_to_mutate = position_to_mutate
    self.is_jax_func = is_jax_func

  def __call__(self, *args, **kwargs):
    to_mutate = args[self.position_to_mutate]
    view_value = to_mutate
    if isinstance(to_mutate, View):
      view_value = to_mutate.torch()
      # Convert the target View to a Tensor, and
      # leave the rest args as is. If other args are
      # also View, they will be converted to tensors
      # in the self.functional dispatch.
    env = view_value._env
    if self.is_jax_func:
      view_value, args, kwargs = env.t2j_iso((view_value, args, kwargs))
      new_value_jax = self.functional(view_value, *args[1:], **kwargs)
      new_value = env.j2t_iso(new_value_jax)
    else:
      new_value = self.functional(view_value, *args[1:], **kwargs)

    if isinstance(to_mutate, View):
      to_mutate.update(new_value)
    else:
      if self.replace:
        to_mutate._elem = new_value._elem
      else:
        to_mutate.copy_(new_value)
    return to_mutate


class OutVariant:

  def __call__(self, *args, **kwargs):
    to_mutate = kwargs['out']
    del kwargs['out']
    to_mutate._elem = self.functional(*args, **kwargs)._elem
    return to_mutate


P = ParamSpec('P')


def convert_dtype(use_default_dtype: bool = True):
  """Converts `dtype` kwarg of function from torch to JAX.

  Args:
    use_default_dtype: Whether to use torch default dtype if none is provided.

  Returns:
    A decorator that wraps a JAX implementation of a torch function.
  """

  def decorator(func: types.TorchCallable):

    @functools.wraps(func)
    def wrapper(*args: P.args,
                dtype: Optional[torch.dtype] = None,
                **kwargs: P.kwargs):
      if not dtype and use_default_dtype:
        dtype = torch.get_default_dtype()
      if isinstance(dtype, torch.dtype):
        jax_dtype = mappings.t2j_dtype(dtype)
      else:
        jax_dtype = dtype

      return func(*args, dtype=jax_dtype, **kwargs)

    return wrapper

  return decorator


def maybe_convert_constant_dtype(val: Optional[types.JaxValue],
                                 dtype: Optional[jnp.dtype]):
  """Optionally converts scalar constant's dtype using `numpy`

  Use in cases where you require a constant and can't handle a traced array.
  """
  if val and dtype:
    if isinstance(val, jax.Array):
      return maybe_convert_constant_dtype(val.item(), dtype)

    return np.array(val, dtype)

  return val


def promote_int_input(f: Callable[Concatenate[jax.Array, P], types.JaxValue]):
  """If the first argument is an int array, promote it to float32."""

  @functools.wraps(f)
  def wrapper(x: jax.Array, *args: P.args, **kwargs: P.kwargs):
    if x.dtype in [jnp.int8, jnp.int16, jnp.int32, jnp.int64]:
      x = x.astype(mappings.t2j_dtype(torch.get_default_dtype()))

    return f(x, *args, **kwargs)

  return wrapper


def foreach_loop(seq: jax.Array,
                 fn: Callable[[jax.Array, jax.Array], jax.Array],
                 init_val=0.0):
  """Run `fn` for each element of 1D array `seq`.

  Similar to `functools.reduce`, but implemented with `jax.lax.fori_loop`."""
  assert len(seq.shape) == 1
  return jax.lax.fori_loop(0, len(seq), lambda i, carry: fn(carry, seq[i]),
                           init_val)
