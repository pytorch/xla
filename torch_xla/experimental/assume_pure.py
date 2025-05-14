from torch_xla._internal.jax_workarounds import requires_jax
import torch_xla.core.xla_builder as xb


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
    `torch.einsum`, `torch.nn.functional.layer_norm`, and a few PyTorch/XLA operators:
    * `torch_xla.experimental.assume_pure` (recursive `assume_pure`)
    * `torch_xla.distributed.spmd.mark_sharding`

  - Other custom PyTorch/XLA operations such as `flash_attention` are not
    supported. This limitation may be lifted in the future.
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
