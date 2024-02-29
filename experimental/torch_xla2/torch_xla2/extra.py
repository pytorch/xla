import jax
import functools
from torch.utils import _pytree as pytree
from torch_xla2 import tensor


def call_jax(jax_function, *args, **kwargs):
    # args, kwargs are torch tensors
    # return val is torch tensor
    args, kwargs = tensor.unwrap((args, kwargs))
    res = jax_function(*args, **kwargs)
    return tensor.wrap(res)


def call_torch(torch_function, *args, **kwargs):
    # args, kwargs are torch tensors
    # return val is torch tensor
    args, kwargs = tensor.wrap((args, kwargs))
    res = torch_function(*args, **kwargs)
    return tensor.unwrap(res)


def fori_loop(lower, upper, body_fn, init_val, *, unroll=None):
    """Torch fori_loop mimicking jax behavior.

    Args:
        lower: lower bound
        upper: upperbound
        init_val: init value (tree of torch.Tensors)
        body_fn is a function that takes (int, a) -> a
            where a is a pytree with torch.Tensors
        unroll = False | True | int
    """
    jax_body = functools.partial(call_torch, body_fn)
    return call_jax(
        jax.lax.fori_loop, 
        lower, 
        upper, 
        jax_body, 
        init_val, 
        unroll=unroll)
