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
    a = torch.randn((3, 3), device=dev, requires_grad=True)
    b = torch.randn((3, 3), device=dev, requires_grad=True)

    ##############pytorch function##############
    def f_torch(a, b):
      return torch.sum(torch.cos(a) + torch.sin(b))

    def f_backward_torch(f, a, b):
      out = f(a, b)
      print(out)
      out.backward()
      return a.grad, b.grad

    _a = a.clone().requires_grad_(True)
    _b = b.clone().requires_grad_(True)
    out_torch = f_torch(_a, _b)

    _a.retain_grad()
    _b.retain_grad()
    out_grad_torch = f_backward_torch(f_torch, _a, _b)

    torch_xla.sync()
    out_torch.detach()
    out_grad_torch = [g.detach() for g in out_grad_torch]

    ##############jax function##############
    def f_jax(a, b):
      import jax.numpy as jnp
      return jnp.sum(jnp.cos(a) + jnp.sin(b))

    # TODO: JAX optimizes a's grad as constant directly and result in error in
    # HLO. Test the following function f_fail_jax once
    # https://github.com/pytorch/xla/issues/8794 is fixed.
    # def f_fail_jax(a, b):
    #   import jax.numpy as jnp
    #   return jnp.sum(a + jnp.sin(b))

    def grad_f_jax(a, b):
      # TODO: Try import jax outside once
      # https://github.com/pytorch/xla/issues/8793 is fixed.
      import jax
      grad_func = jax.grad(f_jax, argnums=(0, 1))
      return grad_func(a, b)

    # TODO: pass (f_jax, a, b) as arguments instead hardcoding f_jax in
    # grad_f_jax once https://github.com/pytorch/xla/issues/8795 is fixed.
    out_jax = xb.call_jax(f_jax, (a, b), {}, 'my_jax_test')
    out_grad_jax = xb.call_jax(grad_f_jax, (a, b), {}, 'my_jax_grad_test')
    torch_xla.sync()

    # forward should produce same output
    torch.testing.assert_close(out_torch, out_jax)
    # backward should produce same gradiant
    torch.testing.assert_close(out_grad_torch, out_grad_jax)

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


if __name__ == "__main__":
  absltest.main()
