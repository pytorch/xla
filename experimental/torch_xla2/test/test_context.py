import unittest

import torch
import torch_xla2
from torch_xla2 import tensor
import torch_xla2.interop

xla_env = tensor.Environment()


class TestContext(unittest.TestCase):

  def setUp(self):
    self.old_var = xla_env.config.use_torch_native_for_cpu_tensor
    xla_env.config.use_torch_native_for_cpu_tensor = False

  def tearDown(self):
    xla_env.config.use_torch_native_for_cpu_tensor = self.old_var

  def test_mode_context_manager(self):
    with xla_env:
      x = torch.full((3, 3), -1)
      self.assertIsInstance(x, tensor.XLATensor2)
      y = x.abs()
      self.assertIsInstance(y, tensor.XLATensor2)

  @staticmethod
  @xla_env
  def _test_mode_decorator():
    x = torch.full((3, 3), -1)
    y = x.abs()

    return x, y

  def test_mode_decorator(self):
    x, y = self._test_mode_decorator()
    self.assertIsInstance(x, tensor.XLATensor2)
    self.assertIsInstance(y, tensor.XLATensor2)

  def test_same_manual_seed(self):
    with xla_env:
      torch.manual_seed(1234)
      x = torch.randn((3, 3))
      self.assertIsInstance(x, tensor.XLATensor2)

      torch.manual_seed(1234)
      y = torch.randn((3, 3))
      self.assertIsInstance(y, tensor.XLATensor2)

    self.assertTrue(torch.equal(torch_xla2.tensor.j2t(x._elem), torch_xla2.tensor.j2t(y._elem)))

  def test_different_manual_seed(self):
    with xla_env:
      torch.manual_seed(1234)
      x = torch.randn((3, 3))
      self.assertIsInstance(x, tensor.XLATensor2)

      torch.manual_seed(12345)
      y = torch.randn((3, 3))
      self.assertIsInstance(y, tensor.XLATensor2)

    self.assertFalse(torch.equal(torch_xla2.tensor.j2t(x._elem), torch_xla2.tensor.j2t(y._elem)))

  def test_jit_with_rng(self):
    @xla_env
    def random_op():
      x = torch.randn(3, 3)
      y = torch.randn(3, 3)
      return x @ y

    random_jit = torch_xla2.interop.jax_jit(random_op)
    self.assertIsInstance(random_jit(), tensor.XLATensor2)

    # Result always expected to be the same for a jitted function because seeds
    # are baked in
    torch.testing.assert_close(
        torch_xla2.tensor.j2t(random_jit()._elem),
        torch_xla2.tensor.j2t(random_jit()._elem),
        atol=0,
        rtol=0)

  def test_generator_seed(self):
    with xla_env:
      x = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))
      y = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))

    # Values will be different, but still check device, layout, dtype, etc
    torch.testing.assert_close(
        torch_xla2.tensor.j2t(x._elem), torch_xla2.tensor.j2t(y._elem))

  def test_buffer(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        c = torch.rand(2)
        self.register_buffer('c', c)
        self.register_buffer('c2', c, persistent=False)

    # Test context manager.
    with xla_env:
      m = M()
      self.assertIsInstance(m.c, tensor.XLATensor2)
      self.assertIsInstance(m.c2, tensor.XLATensor2)
    # Test `to_xla`.
    m = M()
    m = xla_env.to_xla(m)
    self.assertIsInstance(m.c, tensor.XLATensor2)
    self.assertIsInstance(m.c2, tensor.XLATensor2)


if __name__ == "__main__":
  unittest.main()
