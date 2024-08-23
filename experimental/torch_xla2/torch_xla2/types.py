from typing import Callable, Any, Union
import torch
import jax
import jax.numpy as jnp
import sys

if sys.version_info < (3, 10):
  from typing_extensions import ParamSpec, TypeAlias
else:
  from typing import ParamSpec, TypeAlias

P = ParamSpec('P')

TorchValue: TypeAlias = Union[torch.Tensor, torch.dtype, 'TorchCallable', Any]
TorchCallable: TypeAlias = Callable[P, TorchValue]
JaxValue: TypeAlias = Union[jax.Array, jnp.dtype, 'JaxCallable', Any]
JaxCallable: TypeAlias = Callable[P, JaxValue]