import jax
import jax.numpy as jnp
import functools
import torch
from torch.utils import _pytree as pytree
from torch_xla2 import tensor

def torch_view(t):
    # t is an object from jax land
    # view it as-if it's a torch land object
    if isinstance(t, jax.Array):
        return tensor.XLATensor2(t)
    if isinstance(t, type(jnp.int32)):
        return tensor.t2j_type(t)
    if callable(t):
        def new_t(*args, **kwargs):
            # args, kwargs are torch-land 
            args, kwargs = pytree.tree_map(jax_view, (args, kwargs))
            # now they are objs in jax-land
            res = t(*args, **kwargs) # t is jax callable
            # res is jax-land obj
            return pytree.tree_map(torch_view, res)
        return new_t
    # regular types are not changed
    return t


def jax_view(t):
    # t is an object from torch land
    # view it as-if it's a jax land object
    if isinstance(t, torch.Tensor):
        assert isinstance(t, tensor.XLATensor2)
        return t.jax()
    if isinstance(t, type(torch.int32)):
        return tensor.j2t_dtype(t)
    if callable(t):
        def new_t(*args, **kwargs):
            # args, kwargs are jax-land
            args, kwargs = pytree.tree_map(torch_view, (args, kwargs))
            # now they are objs in torch-land
            res = t(*args, **kwargs)
            # res is torch-land obj
            return pytree.tree_map(jax_view, res)
        return new_t
    # regular types are not changed
    return t

def call_jax(jax_func, *args, **kwargs):
    return torch_view(jax_func)(*args, **kwargs)


def call_torch(torch_func, *args, **kwargs):
    return jax_view(torch_func)(*args, **kwargs)


fori_loop = torch_view(jax.lax.fori_loop)

def jax_jit(torch_function, kwargs_for_jax_jit=None):
    kwargs_for_jax_jit = kwargs_for_jax_jit or {}
    jax_func = jax_view(torch_function)
    jitted = jax.jit(jax_func, **kwargs_for_jax_jit)
    return torch_view(jitted)
