import unittest
import torch
import torchax
from flax import linen as nn
from torchax.flax import FlaxNNModule
from torchax.interop import jax_jit
import jax.numpy as jnp
import jax


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


class FlaxTest(unittest.TestCase):

  def test_flax_simple(self):
    flax_model = CNN()

    inputs = jnp.ones((1, 28, 28, 1))
    env = torchax.default_env()
    state = flax_model.init(env.prng_key, inputs)
    expected = flax_model.apply(state, inputs)

    env = torchax.default_env()
    nn_module = FlaxNNModule(env, flax_model, (inputs,), {})
    res = nn_module.forward(inputs)

    self.assertTrue(jnp.allclose(res.jax(), expected))

  def test_flax_functional_call(self):
    flax_model = CNN()

    inputs = jnp.ones((1, 28, 28, 1))
    env = torchax.default_env()
    state = flax_model.init(env.prng_key, inputs)
    expected = flax_model.apply(state, inputs)

    env = torchax.default_env()
    nn_module = FlaxNNModule(env, flax_model, (inputs,), {})

    @jax_jit
    def jitted(weights, args):
      return torch.func.functional_call(nn_module, weights, args)

    with env:
      inputs_torch = torch.ones((1, 28, 28, 1), device='jax')
      state_dict = nn_module.state_dict()
      res = jitted(state_dict, inputs_torch)
      self.assertTrue(jnp.allclose(res.jax(), expected))

  def test_flax_module_nested(self):
    env = torchax.default_env()

    class Parent(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(28, 28)
        sample_cnn_inputs = torch.ones((1, 28, 28, 1), device='jax')
        self.cnn = FlaxNNModule(env, CNN(), (sample_cnn_inputs,), {})

      def forward(self, x):
        y = self.a(x)
        y = y.reshape((-1, 28, 28, 1))
        res = self.cnn(y)
        return res

    with env:
      nn_module = Parent().to('jax')

      @jax_jit
      def jitted(weights, args):
        return torch.func.functional_call(nn_module, weights, args)

      inputs_torch = torch.ones((1, 28, 28), device='jax')
      state_dict = nn_module.state_dict()
      res = jitted(state_dict, inputs_torch)
      print(res)


if __name__ == '__main__':
  unittest.main()
