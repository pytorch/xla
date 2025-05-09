from torch_xla._internal.jax_workarounds import requires_jax
import torch_xla.core.xla_builder as xb
from functools import wraps
import torch
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten, tree_iter, PyTree
import torch_xla

_XLA_COMPUTATION_CACHE = {}

@requires_jax
def assume_pure(fn):
  """Decorates a pure PyTorch/XLA function to skip expensive re-tracing.

  Returns a new function that will only be traced once for each unique
  input tensor shapes or non-tensor input argument values. This is useful
  for removing Lazy Tensor tracing overhead.
  
  The decorated function must be pure (i.e. no side-effects, behavior
  only depends on inputs).

  Limitations:
  - The decorated function can only use upstream PyTorch operators e.g.
    `torch.einsum`, `torch.nn.functional.layer_norm`. Custom PyTorch/XLA
    operations such as `mark_sharding` are not supported. This limitation
    may be lifted in the future.
  """
  from torchax.interop import jax_view
  return j2t_autograd(jax_view(fn))


@requires_jax
def j2t_autograd(fn):
  """Given a JAX function, returns a PyTorch autograd function implemented with `jax.vjp(fn)`.

  It wraps `fn` with `jax.vjp` to compute both the output and residuals (intermediate
  activations). The wrapped function is then run via `call_jax` and integrated into
  the PyTorch autograd framework by saving the residuals into the context object.
  """
  import torchax.interop
  return torchax.interop.j2t_autograd(
      fn, call_jax=lambda fn, *args: xb.call_jax(fn, args))


def assume_pure_torch(use_cache=False):
  """Decorator to mark a function as pure for PyTorch/XLA.

  This decorator builds an XLA computation from the function and caches it.
  The decorated function must be pure (i.e. no side-effects, behavior
  only depends on inputs). 
  Args:
    fn: The function to be decorated.
    use_cache: If True, caches the XLA computation for the function with
      the same name as the function. It is the user's responsibility to ensure
      that the function is called with the same input shapes and types each time
      when using this.

  NOTE: This decorator only works for forward pass. Using it with torch autograd
  will lead to undefined behavior.
  """
  def wrapper(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
      global _XLA_COMPUTATION_CACHE
      # if fn exists in the cache, return the cached result


      # else get the hlo from the function, build XLA computation, cache the
      # computation and return the result of calling the computation
      def make_fake_tensor(v: torch.Tensor) -> torch.Tensor:
        t = xb.create_placeholder_tensor(v.shape, v.dtype)
        return t.requires_grad_(v.requires_grad)

      fake_args = tree_map(make_fake_tensor, args)
      fake_kwargs = tree_map(make_fake_tensor, kwargs)
      
      # TODO: Decide what to include in the cache key.
      if use_cache and _XLA_COMPUTATION_CACHE.get(fn.__name__, None) is not None:
        print(f"Using cached computation for {fn.__name__}")
        fn_computation, output_tree_spec = _XLA_COMPUTATION_CACHE[fn.__name__]
      else:
        fake_outputs = fn(*fake_args, **fake_kwargs)
        fake_outputs, output_tree_spec = tree_flatten(fake_outputs)

        fn_ctx = torch_xla._XLAC.lowering.LoweringContext("FnComputation")
        fn_ctx.set_name_string("fn_ctx")
        fn_ctx.build(fake_outputs)
        fn_hlo = fn_ctx.hlo()

        fn_computation = xb.computation_from_module_proto(f"xla::xb_computation_{fn.__name__}", fn_hlo)
        if use_cache:
          _XLA_COMPUTATION_CACHE[fn.__name__] = (fn_computation, output_tree_spec)

      result = torch_xla._XLAC._xla_user_computation(f"xla::xb_computation_{fn.__name__}",
                                                    args,
                                                    fn_computation)
      result_tree = tree_unflatten(result, output_tree_spec)
      return result_tree
    return inner
  return wrapper