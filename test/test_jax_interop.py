from absl.testing import absltest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


class TestJaxInterop(absltest.TestCase):

  def test_call_jax(self):
    """Test that we can call a JAX function from PyTorch/XLA lazy tensor tracing."""

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
    """Test that call_jax works with PyTree inputs."""

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

  def test_call_jax_some_arg_unused(self):
    """Test when the jax function doesn't use some input arguments."""

    dev = xm.xla_device()
    a = torch.randn((3, 3), device=dev)
    b = torch.randn((3, 3), device=dev)
    c = torch.randn((3, 3), device=dev)
    d = torch.randn((3, 3), device=dev)

    def f(a, b, c, d):
      import jax.numpy as jnp
      return a + jnp.sin(b)

    o = xb.call_jax(f, (a, b, c, d), {}, 'my_test')
    torch_xla.sync()
    torch.testing.assert_close(o, a + torch.sin(b), check_device=False)

  def test_call_jax_grad(self):
    """Test calling a simple jax.grad transformed function."""

    dev = xm.xla_device()
    a = torch.randn((3, 3), device=dev, requires_grad=True)
    b = torch.randn((3, 3), device=dev, requires_grad=True)
    torch_xla.sync()

    import jax

    def f_torch(a, b):
      return torch.sum(a + torch.sin(b))

    def f_backward_torch(f, a, b):
      out = f(a, b)
      out.backward()
      return a.grad, b.grad

    def f_jax(a, b):
      import jax.numpy as jnp
      # JAX optimizes a's grad as constant, so it will never use `a`.
      # We should support that.
      return jnp.sum(a + jnp.sin(b))

    grad_f_jax = jax.grad(f_jax, argnums=(0, 1))

    out_torch = f_torch(a, b)
    out_grad_torch = f_backward_torch(f_torch, a, b)
    out_jax = xb.call_jax(f_jax, (a, b), {})
    out_grad_jax = xb.call_jax(grad_f_jax, (a, b), {})
    torch_xla.sync()

    # forward should produce same output
    torch.testing.assert_close(out_torch, out_jax)
    # backward should produce same gradient
    torch.testing.assert_close(out_grad_torch, out_grad_jax)


if __name__ == "__main__":
  absltest.main()
