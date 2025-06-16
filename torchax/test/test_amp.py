import unittest
import jax
import jax.numpy as jnp
import torchax
from torchax import interop
import torch


class AutocastTest(unittest.TestCase):

  def setUp(self):
    self.env = torchax.default_env()

  def test_auto_cast_ir(self):
    with self.env:
      with torchax.amp.autocast('jax', dtype=torch.bfloat16, env=self.env):
        a = jax.ShapeDtypeStruct((2, 2), jnp.float32)
        b = jax.ShapeDtypeStruct((2, 2), jnp.float32)
        ir_text = jax.jit(interop.jax_view(torch.matmul)).lower(a, b).as_text()
        self.assertIn('tensor<2x2xbf16>', ir_text)

  def test_auto_cast_matmul(self):
    with self.env:
      a = torch.randn(2, 2, device='jax')
      b = torch.randn(2, 2, device='jax')
      with torchax.amp.autocast('jax', dtype=torch.bfloat16, env=self.env):
        c = a @ b

      self.assertEqual(c.dtype, torch.bfloat16)

      with torch.autocast('cpu', dtype=torch.bfloat16):
        c_cpu = a.cpu() @ b.cpu()

      self.assertTrue(torch.allclose(c.cpu(), c_cpu))


if __name__ == '__main__':
  unittest.main()
