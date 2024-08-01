from typing import TypeAlias, Callable, ParamSpec, Any, Union
import torch
import jax
import jax.numpy as jnp


P = ParamSpec('P')

TorchValue: TypeAlias = Union[torch.Tensor, torch.dtype, 'TorchCallable', Any]
TorchCallable: TypeAlias = Callable[P, TorchValue]
JaxValue: TypeAlias = Union[jax.Array, jnp.dtype, 'JaxCallable', Any]
JaxCallable: TypeAlias = Callable[P, JaxValue]