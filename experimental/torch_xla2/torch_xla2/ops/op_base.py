import functools
import torch
from torch_xla2 import interop, tensor
from torch_xla2 import types

from typing import Callable, Optional, ParamSpec, Sequence


class BinaryOpWithPromotion:

    def __init__(self, inner):
        self.inner = inner 

    def _get_dtype(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.dtype
        else:
            if isinstance(obj, float):
                return torch.float32
            if isinstance(obj, int):
                return torch.int32
        return torch.float32


    def __call__(self, *args, **kwargs):
        # args are torch.Tensor
        res = interop.torch_view(self.jax)(*args, **kwargs)

        dtype = torch.promote_types(
            self._get_dtype(args[0]), 
            self._get_dtype(args[1]))
        if dtype != res.dtype:
            res = res.to(dtype)
        return res


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
      jax_dtype = tensor.t2j_dtype(dtype)

      return func(*args, dtype=jax_dtype, **kwargs)

    return wrapper

  return decorator
