import functools
import torch
import jax
import jax.numpy as jnp
from jax import tree_util as pytree
from torch_xla2 import tensor
import torch_xla2

from torch_xla2.types import JaxValue, TorchValue, JaxCallable, TorchCallable



def _torch_view(t: JaxValue) -> TorchValue:
    # t is an object from jax land
    # view it as-if it's a torch land object
    if isinstance(t, jax.Array):
        # TODO
        return tensor.XLATensor2(t, torch_xla2.default_env())
    if isinstance(t, type(jnp.int32)):
        return tensor.t2j_type(t)
    if callable(t):  # t is a JaxCallable
        return functools.partial(call_jax, t)
    # regular types are not changed
    return t

torch_view = functools.partial(pytree.tree_map, _torch_view)


def _jax_view(t: TorchValue) -> JaxValue:
    # t is an object from torch land
    # view it as-if it's a jax land object
    if isinstance(t, torch.Tensor):
        assert isinstance(t, tensor.XLATensor2)
        return t.jax()
    if isinstance(t, type(torch.int32)):
        return tensor.j2t_dtype(t)

    # torch.nn.Module needs special handling
    if not isinstance(t, torch.nn.Module) and callable(t):  # t is a TorchCallable
        return functools.partial(call_torch, t)
    # regular types are not changed
    return t

jax_view = functools.partial(pytree.tree_map, _jax_view)


def call_jax(jax_func: JaxCallable, 
             *args: TorchValue, 
             **kwargs: TorchValue) -> TorchValue:
    args, kwargs = jax_view((args, kwargs))
    res: JaxValue = jax_func(*args, **kwargs)
    return torch_view(res)


def call_torch(torch_func: TorchCallable, *args: JaxValue, **kwargs: JaxValue) -> JaxValue:
    args, kwargs = torch_view((args, kwargs))
    with torch_xla2.default_env():
        res: TorchValue = torch_func(*args, **kwargs)
    return jax_view(res)


fori_loop = torch_view(jax.lax.fori_loop)

def jax_jit(torch_function, kwargs_for_jax_jit=None):
    kwargs_for_jax_jit = kwargs_for_jax_jit or {}
    jax_func = jax_view(torch_function)
    jitted = jax.jit(jax_func, **kwargs_for_jax_jit)
    return torch_view(jitted)

