"""If you don't know which file a test should go, and don't want to make a new file
for a small test. PUt it here
"""
import torch
import unittest
import torchax
import jax
import jax.numpy as jnp


class MiscTest(unittest.TestCase):

  def test_extract_jax_kwargs(self):

    class M(torch.nn.Module):

      def forward(self, a, b):
        return torch.sin(a) + torch.cos(b)

    weights, func = torchax.extract_jax(M())
    res = func(
        weights,
        args=(),
        kwargs={
            'a': jnp.array([1, 2, 3]),
            'b': jnp.array([3, 4, 5])
        })
    self.assertTrue(
        jnp.allclose(
            res,
            jnp.sin(jnp.array([1, 2, 3])) + jnp.cos(jnp.array([3, 4, 5]))))

  def test_to_device(self):
    env = torchax.default_env()
    with env:
      step1 = torch.ones(
          100,
          100,
      )
      step2 = torch.triu(step1, diagonal=1)
      step3 = step2.to(dtype=torch.bool, device='jax')
      self.assertEqual(step3.device.type, 'jax')

  def test_to_device_twice(self):
      env = torchax.default_env()
      env.config.debug_print_each_op = True
      with env:
        step1 = torch.ones(
            100,
            100,
        )
        step2 = torch.triu(step1, diagonal=1)
        step3 = step2.to(dtype=torch.bool, device='jax')
        step3.to('jax')
        self.assertEqual(step3.device.type, 'jax')


if __name__ == '__main__':
  unittest.main()
