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

  Limitations:

  - The decorated function must be pure (i.e. no side-effects, behavior
    only depends on inputs).
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
  activations). The wrapped function is then jitted via `xb.call_jax` and integrated into
  the PyTorch autograd framework by saving the residuals into the context object.
  """

  @wraps(fn)
  def inner(*args, **kwargs):
    from jax.tree_util import tree_flatten, tree_unflatten
    from jax.util import safe_zip

    class JaxFun(torch.autograd.Function):

      @staticmethod
      def forward(ctx, tree_def, *flat_inputs):
        # Reconstruct the original args and kwargs
        args, kwargs = tree_unflatten(tree_def, flat_inputs)

        # Execute the JAX computation
        # Pass the reconstructed args/kwargs tuple as the primal
        y, fun_vjp = xb.call_jax(
            _jax_forward, args=(
                fn,
                (args, kwargs),
            ))

        # Save necessary information for backward
        # Flatten the vjp function. `vjp_spec` contains a jaxpr for the backward pass.
        # `residuals` contains the tensors needed for the backward pass.`
        residuals, vjp_spec = tree_flatten(fun_vjp)
        ctx.vjp_spec = vjp_spec
        ctx.save_for_backward(*residuals)

        return y

      @staticmethod
      def backward(ctx, *grad_out):
        assert len(grad_out) > 0
        grad_out = grad_out if len(grad_out) > 1 else grad_out[0]

        input_grads_structured = xb.call_jax(
            _jax_backward, args=(ctx.vjp_spec, ctx.saved_tensors, grad_out))

        # Flatten the gradients to match the flat inputs to forward
        flat_input_grads, _ = tree_flatten(input_grads_structured)

        # Construct the gradient tuple to be returned.
        # It needs to match the inputs to forward: (tree_def, *flat_inputs)
        # The first gradient (for tree_def) is None.
        # The subsequent gradients correspond to flat_inputs.
        # We need to put a None for inputs that did not require gradients.
        final_grads = [None]
        for needs_grad, grad in safe_zip(ctx.needs_input_grad[1:],
                                         flat_input_grads):
          final_grads.append(grad if needs_grad else None)

        return tuple(final_grads)

    sig = signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    flat_args_kwargs, tree_def = tree_flatten((bound.args, bound.kwargs))
    return JaxFun.apply(tree_def, *flat_args_kwargs)

  return inner


def _jax_forward(fn, primals):
  """JAX function to compute output and vjp function.

  primals should be a tuple (args, kwargs).
  """
  import jax

  def fn_wrapper(a, kw):
    return fn(*a, **kw)

  return jax.vjp(fn_wrapper, *primals)


def _jax_backward(vjp_spec, saved_tensors, grad_out):
  """JAX function to compute input gradients.

  Unflattening `saved_tensors` with `vjp_spec` should restore the original vjp function.
  """
  from jax.tree_util import tree_unflatten
  fun_vjp = tree_unflatten(vjp_spec, saved_tensors)
  return fun_vjp(grad_out)
