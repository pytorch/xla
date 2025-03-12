import weakref
from absl.testing import absltest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


class TestJaxInterop(absltest.TestCase):

  def test_call_jax(self):
    """
    Test that we can call a JAX function from PyTorch/XLA lazy tensor tracing.
    """

    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, b):
      import jax.numpy as jnp
      return a + jnp.sin(b)

    b = xb.call_jax(f, (a, a), {}, 'my_test')
    torch_xla.sync()
    torch.testing.assert_close(
        b, torch.sin(torch.ones(3, 3)) + 1, check_device=False)

  def test_call_jax_pytree(self):
    """
    Test that call_jax works with PyTree inputs.
    """
    dev = xm.xla_device()
    a = torch.ones((2, 2), device=dev)
    b = torch.ones((2, 2), device=dev) * 2

    def f(inputs):
      a = inputs['a']
      b = inputs['b']
      return a @ b

    inputs = {'a': a, 'b': b}
    c = xb.call_jax(f, (inputs,))
    torch_xla.sync()
    torch.testing.assert_close(
        c,
        torch.tensor(
            [
                [4, 4],
                [4, 4],
            ],
            dtype=torch.float32,
        ),
        check_device=False)

  def test_call_jax_avoid_repeated_tracing(self):
    """
    Test that repeatedly calling `call_jax` with the same function does
    not lead to repeated tracing and jitting, which would be detrimental
    to performance.
    """

    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, b):
      return a + b

    # See documentation on `_fn_flattened_inputs` for this attribute.
    starting_trace_count = getattr(xb._fn_flattened_inputs, '_num_traces', 0)
    starting_cached_hlo_count = len(xb._JAX_HLO_CACHE)
    for _ in range(10):
      a = xb.call_jax(f, (a, a))
      torch_xla.sync()
    ending_trace_count = getattr(xb._fn_flattened_inputs, '_num_traces', 0)
    ending_cached_hlo_count = len(xb._JAX_HLO_CACHE)

    # `f` is only traced once.
    self.assertEqual(starting_trace_count + 1, ending_trace_count)
    self.assertEqual(starting_cached_hlo_count + 1, ending_cached_hlo_count)

    # Now let's define a different function and trace that.
    def g(a, b):
      return a * b

    a = xb.call_jax(g, (a, a))
    torch_xla.sync()
    new_trace_count = getattr(xb._fn_flattened_inputs, '_num_traces', 0)
    new_cached_hlo_count = len(xb._JAX_HLO_CACHE)
    self.assertEqual(starting_trace_count + 2, new_trace_count)
    self.assertEqual(starting_cached_hlo_count + 2, new_cached_hlo_count)


if __name__ == "__main__":
  absltest.main()
