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


if __name__ == "__main__":
  absltest.main()
