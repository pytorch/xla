import torch
import jax
import jax.numpy as jnp

import dataclasses
from typing import TypeAlias, Callable, ParamSpec, Any, Union, Dict

P = ParamSpec('P')

TorchValue: TypeAlias = Union[torch.Tensor, torch.dtype, 'TorchCallable', Any]
TorchCallable: TypeAlias = Callable[P, TorchValue]

JaxValue: TypeAlias = Union[jax.Array, jnp.dtype, 'TorchCallable', Any]
JaxCallable: TypeAlias = Callable[P, JaxValue]


@dataclasses.dataclass
class Operator:
    torch_op: TorchCallable
    func: Union[TorchCallable, JaxCallable]
    is_jax_function: bool
    is_user_defined: bool
    needs_env: bool


all_aten_ops: Dict[TorchCallable, Operator] = {}
all_torch_functions: Dict[TorchCallable, Operator] = {}


def register_torch_dispatch_op(
    aten_op, impl_callable, 
    is_jax_function=True, 
    is_user_defined=False,
    needs_env=False,
):
    op = Operator(
        aten_op, impl_callable, 
        is_jax_function=is_jax_function,
        is_user_defined=is_user_defined,
        needs_env=needs_env)
    all_aten_ops[aten_op] = op
    return impl_callable 


def register_torch_function_op(
    torch_func, impl_callable, 
    is_jax_function=True, 
    is_user_defined=False,
    needs_env=False,
):
    op = Operator(
        torch_func, impl_callable, 
        is_jax_function=is_jax_function,
        is_user_defined=is_user_defined,
        needs_env=needs_env)
    all_torch_functions[torch_func] = op
    return impl_callable 