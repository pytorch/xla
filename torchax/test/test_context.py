import unittest

import torch
import torchax
from torchax import tensor
import torchax.interop

xla_env = torchax.default_env()


class TestContext(unittest.TestCase):

  def test_mode_context_manager(self):
    with xla_env:
      x = torch.full((3, 3), -1, device='jax')
      self.assertIsInstance(x, tensor.Tensor)
      y = x.abs()
      self.assertIsInstance(y, tensor.Tensor)

  @staticmethod
  @xla_env
  def _test_mode_decorator():
    x = torch.full((3, 3), -1).to('jax')
    y = x.abs()

    return x, y

  def test_mode_decorator(self):
    x, y = self._test_mode_decorator()
    self.assertIsInstance(x, tensor.Tensor)
    self.assertIsInstance(y, tensor.Tensor)

  def test_same_manual_seed(self):
    with xla_env:
      xla_env.manual_seed(1234)
      x = torch.randn((3, 3), device='jax')
      self.assertIsInstance(x, tensor.Tensor)

      xla_env.manual_seed(1234)
      y = torch.randn((3, 3), device='jax')
      self.assertIsInstance(y, tensor.Tensor)

      self.assertTrue(torch.allclose(x, y))

  def test_different_manual_seed(self):
    with xla_env:
      xla_env.manual_seed(1234)
      x = torch.randn((3, 3), device='jax')
      self.assertIsInstance(x, tensor.Tensor)

      xla_env.manual_seed(12345)
      y = torch.randn((3, 3), device='jax')
      self.assertIsInstance(y, tensor.Tensor)

      self.assertFalse(torch.allclose(x, y))

  def test_jit_with_rng(self):

    with xla_env:

      def random_op():
        x = torch.randn(3, 3, device='jax')
        y = torch.randn(3, 3, device='jax')
        return x @ y

      random_jit = torchax.interop.jax_jit(random_op)
      self.assertIsInstance(random_jit(), tensor.Tensor)

      # If we run the JIT twice, the random values should be different.
      # TODO(qihqi): think about API for passing down seed
      #  with self.assertRaises(AssertionError):
      #    torch.testing.assert_close(random_jit(), random_jit(), atol=0, rtol=0)

  def test_generator_seed(self):
    with xla_env:
      x = torch.randn(
          2, 3, generator=torch.Generator().manual_seed(0), device='jax')
      y = torch.randn(
          2, 3, generator=torch.Generator().manual_seed(0), device='jax')

      # Values will be the same given the same seed.
      torch.testing.assert_close(x, y)

  def test_buffer(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        c = torch.rand(2)
        self.register_buffer('c', c)
        self.register_buffer('c2', c, persistent=False)

    # Test context manager.
    with xla_env:
      m = M().to('jax')
      self.assertIsInstance(m.c, tensor.Tensor)
      self.assertIsInstance(m.c2, tensor.Tensor)
    # Test `to_xla`.
    m = M()
    m = xla_env.to_xla(m)
    self.assertIsInstance(m.c, tensor.Tensor)
    self.assertIsInstance(m.c2, tensor.Tensor)


if __name__ == "__main__":
  unittest.main()
