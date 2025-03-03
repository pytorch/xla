from absl.testing import absltest

import torch
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


class TestJaxInterop(absltest.TestCase):

  def test_call_jax(self):
    import jax.numpy as jnp

    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, b):
      return a + jnp.sin(b)

    b = xb.call_jax(f, (a, a), {}, 'hame')
    print(b)


if __name__ == "__main__":
  absltest.main()
