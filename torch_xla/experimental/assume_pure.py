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


# Define the JAX function to compute value and vjp
def _jax_forward(fn, primals):
  import jax

  # Prepare the function call for jax.vjp
  # jax.vjp expects positional primals. We wrap fn to accept args, kwargs.
  def fn_wrapper(a, kw):
    return fn(*a, **kw)

  # primals will be (args_rec, kwargs_rec)
  return jax.vjp(fn_wrapper, *primals)  # Unpack primals here


def _jax_backward(vjp_spec, saved_tensors, grad_args):
  from jax.tree_util import tree_unflatten
  fun_vjp = tree_unflatten(vjp_spec, saved_tensors)
  return fun_vjp(grad_args)


def _jax2torch(fn):

  @wraps(fn)
  def inner(*args, **kwargs):
    from jax.tree_util import tree_flatten, tree_unflatten

    class JaxFun(torch.autograd.Function):

      @staticmethod
      def forward(ctx, tree_def, *flat_args_kwargs_values):
        # Note: flat_args_kwargs_values contains the *values* from the flattened structure

        # Reconstruct the original args and kwargs inside forward
        args_rec, kwargs_rec = tree_unflatten(tree_def, flat_args_kwargs_values)

        # Execute the JAX computation
        # Pass the reconstructed args/kwargs tuple as the primal
        y_, fun_vjp = xb.call_jax(
            _jax_forward, args=(
                fn,
                (args_rec, kwargs_rec),
            ))

        # Save necessary information for backward
        # Flatten the vjp function (may contain tensors/non-tensors)
        residuals, vjp_spec = tree_flatten(fun_vjp)

        # Save only tensors needed for backward (the residuals)
        # Autograd automatically gives None gradients for non-tensor inputs.
        # We need the vjp_spec (non-tensor) and tree_def for reconstruction.
        ctx.vjp_spec = vjp_spec
        ctx.tree_def = tree_def  # Need tree_def to structure gradients in backward
        # Save residuals which might be tensors needed by the VJP function
        ctx.save_for_backward(*residuals)

        # Return the results (potentially nested structure)
        # The user expects the original output structure of fn
        return y_

      @staticmethod
      def backward(ctx, *grad_args):
        assert len(grad_args) > 0
        grad_args = grad_args if len(grad_args) > 1 else grad_args[0]

        input_grads_structured = xb.call_jax(
            _jax_backward, args=(ctx.vjp_spec, ctx.saved_tensors, grad_args))

        # Flatten the gradients to match the flat inputs to forward
        flat_input_grads, _ = tree_flatten(input_grads_structured)

        # Construct the gradient tuple for autograd.
        # It needs to match the inputs to forward: (tree_def, *flat_args_kwargs_values)
        # The first gradient (for tree_def) is None.
        # The following gradients correspond to flat_args_kwargs_values.
        # We need to return None for inputs that did not require gradients.
        final_grads = [None]  # Gradient for tree_def is None
        input_grad_iter = iter(flat_input_grads)
        for i, needs_grad in enumerate(
            ctx.needs_input_grad[1:]):  # Skip ctx for tree_def
          if needs_grad:
            # This input leaf required grad, so JAX should have computed one
            try:
              grad = next(input_grad_iter)
              final_grads.append(grad)
            except StopIteration:
              # Should not happen if JAX computed grads for all required inputs
              raise ValueError(
                  "Warning: Mismatch between required grads and JAX output grads."
              ) from None
          else:
            # This input leaf did not require grad
            final_grads.append(None)
            grad = next(input_grad_iter)

        return tuple(final_grads)

    sig = signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    flat_args_kwargs, tree_def = tree_flatten((bound.args, bound.kwargs))
    return JaxFun.apply(tree_def, *flat_args_kwargs)

  return inner
