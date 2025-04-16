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

  def test_call_jax_input_pytree(self):
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

  def test_call_jax_output_pytree(self):
    """Test that call_jax works with PyTree outputs."""

    dev = xm.xla_device()
    a = torch.ones((2, 2), device=dev)

    def f(a):
      b = a + 1
      c = a + 2
      return {'b': b, 'c': c}

    out = xb.call_jax(f, (a,))
    torch_xla.sync()
    torch.testing.assert_close(
        out['b'],
        torch.tensor(
            [
                [2, 2],
                [2, 2],
            ],
            dtype=torch.float32,
        ),
        check_device=False)
    torch.testing.assert_close(
        out['c'],
        torch.tensor(
            [
                [3, 3],
                [3, 3],
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

  def test_call_jax_non_tensor_args(self):
    """Test that call_jax works with non-tensor arguments."""

    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, num: float, string: str, dictionary: dict, none):
      assert isinstance(string, str)
      import jax.numpy as jnp
      if none is None:
        return a + jnp.sin(num) + int(string) + dictionary['x']
      raise ValueError('none should be None')

    b = xb.call_jax(
        f, (
            a,
            1.0,
            "10",
            {
                "x": torch.tensor(0.25, device=dev)
            },
        ),
        kwargs={"none": None})
    torch_xla.sync()
    torch.testing.assert_close(
        b, torch.sin(torch.ones(3, 3)) + 1 + 10 + 0.25, check_device=False)

  def test_call_jax_cache_hlo(self):
    """Test that the HLO of a jax function should be cached."""

    starting_cache_misses = xb._jax_to_xla_computation_cache_elements()

    # Let's trace two different jax functions a couple of times.
    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, b):
      import jax.numpy as jnp
      return a + jnp.sin(b)

    def g(a, b):
      import jax.numpy as jnp
      return a + jnp.cos(b)

    xb.call_jax(f, (a, a))
    xb.call_jax(f, (a, a))
    xb.call_jax(g, (a, a))
    xb.call_jax(g, (a, a))

    ending_cache_misses = xb._jax_to_xla_computation_cache_elements()
    self.assertEqual(ending_cache_misses - starting_cache_misses, 2)

  def test_call_jax_cache_by_shape(self):
    """Test that the same function may be traced again if the shape of its arguments changes."""

    starting_cache_misses = xb._jax_to_xla_computation_cache_elements()

    # Let's trace the same jax function with different shapes.
    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)
    b = torch.ones((2, 2), device=dev)

    def f(a, b):
      import jax.numpy as jnp
      return a + jnp.sin(b)

    xb.call_jax(f, (a, a))
    xb.call_jax(f, (b, b))

    ending_cache_misses = xb._jax_to_xla_computation_cache_elements()
    self.assertEqual(ending_cache_misses - starting_cache_misses, 2)

  def test_call_jax_cache_by_tree_spec(self):
    """Test that the same function may be traced again if the tree spec of its arguments changes."""
    starting_cache_misses = xb._jax_to_xla_computation_cache_elements()

    # Let's trace the same jax function with different tree specs.
    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)
    b = torch.ones((3, 2), device=dev)

    def f(inputs):
      a = inputs['a']
      b = inputs['b']
      return a @ b

    xb.call_jax(f, ({'a': a, 'b': a},))
    xb.call_jax(f, ({'a': a, 'b': b},))

    ending_cache_misses = xb._jax_to_xla_computation_cache_elements()
    self.assertEqual(ending_cache_misses - starting_cache_misses, 2)

  def test_call_jax_cache_by_static_args(self):
    """Test that the same function may be traced again if a non-tensor argument changes."""
    starting_cache_misses = xb._jax_to_xla_computation_cache_elements()

    # Let's trace the same jax function with different static args.
    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, num: float):
      import jax.numpy as jnp
      return a + jnp.sin(num)

    xb.call_jax(f, (a, 1.0))
    xb.call_jax(f, (a, 2.0))
    xb.call_jax(f, (a, 3.0))

    ending_cache_misses = xb._jax_to_xla_computation_cache_elements()
    self.assertEqual(ending_cache_misses - starting_cache_misses, 3)

  def test_call_jax_with_different_jax_config(self):
    import jax
    starting_cache_misses = xb._jax_to_xla_computation_cache_elements()

    # Let's trace the same jax function with different static args.
    dev = xm.xla_device()
    a = torch.ones((3, 3), device=dev)

    def f(a, num: float):
      import jax.numpy as jnp
      return a + jnp.sin(num)

    jax.config.update('jax_default_matmul_precision', "highest")
    xb.call_jax(f, (a, a))
    jax.config.update('jax_default_matmul_precision', "default")
    xb.call_jax(f, (a, a))

    ending_cache_misses = xb._jax_to_xla_computation_cache_elements()
    self.assertEqual(ending_cache_misses - starting_cache_misses, 2)


if __name__ == "__main__":
  absltest.main()
