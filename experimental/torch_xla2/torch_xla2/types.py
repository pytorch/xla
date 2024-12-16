from typing import Callable, Any, Union, ParamSpec, TypeAlias
import torch
import jax
import jax.numpy as jnp
import sys

P = ParamSpec('P')

TorchValue: TypeAlias = Union[torch.Tensor, torch.dtype, 'TorchCallable', Any]
TorchCallable: TypeAlias = Callable[P, TorchValue]
JaxValue: TypeAlias = Union[jax.Array, jnp.dtype, 'JaxCallable', Any]
JaxCallable: TypeAlias = Callable[P, JaxValue]