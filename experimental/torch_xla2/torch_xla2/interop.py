import copy
import functools
import torch
from torch.nn.utils import stateless as torch_stateless
import jax
import jax.numpy as jnp
from jax import tree_util as pytree
from jax.experimental.shard_map import shard_map
from torch_xla2 import tensor
import torch_xla2

from torch_xla2.types import JaxValue, TorchValue, JaxCallable, TorchCallable


def extract_all_buffers(m: torch.nn.Module):
    buffers = {}
    params = {}
    def extract_one(module, prefix):
        for k in dir(module):
            try:
                v = getattr(module, k)
            except:
                continue
            qual_name = prefix + k
            if isinstance(v, torch.nn.parameter.Parameter) and v.requires_grad:
                params[qual_name] = v
            elif isinstance(v, torch.Tensor):
                buffers[qual_name] = v
        for name, child in module.named_children():
            extract_one(child, prefix + name + '.')
    extract_one(m, '')
    return params, buffers


def set_all_buffers(m, params, buffers):
    def set_one(module, prefix):
        for k in dir(module):
            qual_name = prefix + k
            if (potential_v := buffers.get(qual_name)) is not None:
                setattr(module, k, potential_v)
            elif (potential_v := params.get(qual_name)) is not None:
                print(k, potential_v)
                setattr(module, k, torch.nn.Parameter(potential_v))
        for name, child in module.named_children():
            set_one(child, prefix + name + '.')

    set_one(m, '')


class JittableModule:

    def __init__(self, m: torch.nn.Module):
        self.params, self.buffers = extract_all_buffers(m)
        self._model = m


    def __call__(self, *args, **kwargs):
        res = self._model(*args, **kwargs)
        return res

    def functional_call(
            self, method_name, params, buffers, args, kwargs=None):
        kwargs = kwargs or {}
        params_copy = copy.copy(params)
        params_copy.update(buffers)
        with torch_stateless._reparametrize_module(self._model, params_copy):
            res = getattr(self._model, method_name)(*args, **kwargs)
        return res


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


def wrap_jax_jit(torch_function, jax_jit_func=jax.jit, kwargs_for_jax=None):
    kwargs_for_jax = kwargs_for_jax or {}
    jax_func = jax_view(torch_function)
    jitted = jax_jit_func(jax_func, **kwargs_for_jax)
    return torch_view(jitted)


def jax_jit(torch_function, kwargs_for_jax_jit=None):
    return wrap_jax_jit(torch_function, jax_jit_func=jax.jit,
                        kwargs_for_jax=kwargs_for_jax_jit)


def jax_shard_map(torch_function, kwargs_for_jax_shard_map=None):
    return wrap_jax_jit(torch_function, jax_jit_func=shard_map,
                        kwargs_for_jax=kwargs_for_jax_shard_map)
