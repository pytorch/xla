"""The `jlibrary` module has functions which help to preserve torch.library ops
during export. This includes aten ops, and custom operations.
"""

import torch
import torch.nn as nn
import torchax
from torchax.ops import jaten
import jax
import functools


def _jit_composite_impl(composite_name, jaxpr_impl, **jit_args):
  """Wrap a jaxpr in a jitted function with the proper composite name
  TODO: Wrap JIT in a `stablehlo.composite` op, instead of generating a call op.
  """

  def composite_impl(*args):
    return jaxpr_impl(*args)

  composite_impl.__name__ = composite_name
  composite_impl.__qualname__ = composite_name
  return jax.jit(composite_impl, **jit_args)


def register_jax_composite(composite_name, impl, *ops, **jit_args):
  """Register a composite using a JAX implementation.
    composite_name - The name of the library op to use in the exported composite
    impl           - A JAX lowering for the library operation
    *ops           - Variadic torch.ops to lower using `impl`.
    **jit_args     - Additional parameters to forward to JAX jit.

  This is used to register custom lowerings with an explicit jaxpr
  implementation, such as preserving a specific aten op using a jaten impl.

  For custom torch op registration with a decomposition written in torch,
  use `register_torch_composite`.

  For jit params and troubleshooting see:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
  """

  @jaten.op(*ops)
  def _composite_impl(*args):
    return _jit_composite_impl(composite_name, impl, **jit_args)(*args)


def register_torch_composite(composite_name, impl, *ops, **jit_args):
  """Register a torch decomposition as a composite.
  This is useful for registerring custom torch op libraries as composite ops.

  The `impl` can be the `@impl` used to define the torch custom library op.
  This must be a function or module impl that provides the decompositions, and
  not an instance of the custom op.

  TODO: Better error handling, or can we make this an instance of the op as a param?

  For jit params and troubleshooting see:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
  """

  @jaten.op(*ops)
  def _composite_impl(*args):

    class ImplWrapper(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, *args):
        return impl(*args)

    # Note: avoid refactoring to share code with register_jaxpr_composite.
    # The `extract_jax` call must live in the `@jaten.op` handler. If called
    # outside of the handler, we would build the jaxpr representation of the
    # module once during registration, potentially missing op registrations that
    # come after. I.e. may miss nested abstractions if we build jaxpr AoT.
    state, jfn = torchax.extract_jax(ImplWrapper())
    jaxpr_impl = lambda *args: jfn(state, tuple([*args]))
    return _jit_composite_impl(composite_name, jaxpr_impl, **jit_args)(*args)
