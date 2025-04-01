import torch

from inspect import signature
from functools import wraps

from torch_xla._internal.jax_workarounds import requires_jax
import torch_xla.core.xla_builder as xb


@requires_jax
def assume_pure(fn):
  """Decorates a pure PyTorch/XLA function to skip expensive re-tracing.

  Returns a new function that will only be traced once for each unique
  input tensor shapes or non-tensor input argument values. This is useful
  for removing Lazy Tensor tracing overhead.

  The decorated function must be pure (i.e. no side-effects).
  """
  from torchax.interop import jax_view
  return _jax2torch(jax_view(fn))


def _jax2torch(fn):

  @wraps(fn)
  def inner(*args, **kwargs):
    import jax
    from jax.tree_util import tree_flatten, tree_unflatten

    class JaxFun(torch.autograd.Function):

      @staticmethod
      def forward(ctx, *args):

        def jax_func(fn, *args):
          return jax.vjp(fn, *args)

        y_, fun_vjp = xb.call_jax(jax_func, args=(fn, *args))
        residuals, ctx.vjp_spec = tree_flatten(fun_vjp)
        ctx.save_for_backward(residuals)
        return y_

      @staticmethod
      def backward(ctx, *grad_args):
        fun_vjp = tree_unflatten(ctx.vjp_spec, ctx.saved_tensors)
        assert len(grad_args) > 0
        grad_args = grad_args if len(grad_args) > 1 else grad_args[0]

        def jax_func(fun_vjp, grad_args):
          return fun_vjp(grad_args)

        grads = xb.call_jax(jax_func, args=(fun_vjp, grad_args))
        grads = tuple(
            (t if isinstance(t, torch.Tensor) else None for t in grads))
        return grads

    sig = signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return JaxFun.apply(*bound.arguments.values())

  return inner
