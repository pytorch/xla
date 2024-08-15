import functools
import jax
import jax.numpy as jnp
import torch
from torch_xla2.ops import mappings
from torch_xla2 import types

from typing import Callable, Concatenate, Optional, ParamSpec


class InplaceOp:

    def __init__(self, functional_op, position_to_mutate=0):
        self.functional = functional_op
        self.position_to_mutate = position_to_mutate

    def __call__(self, *args, **kwargs):
        to_mutate = args[0]
        to_mutate.copy_(self.functional(*args, **kwargs))
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
      jax_dtype = mappings.t2j_dtype(dtype)

      return func(*args, dtype=jax_dtype, **kwargs)

    return wrapper

  return decorator


def promote_int_input(f: Callable[Concatenate[jax.Array, P], types.JaxValue]):
   """If the first argument is an int array, promote it to float32."""
   @functools.wraps(f)
   def wrapper(x: jax.Array, *args: P.args, **kwargs: P.kwargs):
      if x.dtype in [jnp.int8, jnp.int16, jnp.int32, jnp.int64]:
        x = x.astype(mappings.t2j_dtype(torch.get_default_dtype()))

      return f(x, *args, **kwargs)

   return wrapper


def foreach_loop(
  seq: jax.Array, fn: Callable[[jax.Array, jax.Array], jax.Array], init_val=0.0
):
  """Run `fn` for each element of 1D array `seq`.

  Similar to `functools.reduce`, but implemented with `jax.lax.fori_loop`."""
  assert len(seq.shape) == 1
  return jax.lax.fori_loop(
    0, len(seq), lambda i, carry: fn(carry, seq[i]), init_val
  )
