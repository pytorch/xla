import jax
import torch
import torch._functorch
from torch_xla2 import tensor


def extract_jax(mod: torch.nn.Module):
  """Returns a pytree of jax.ndarray and a jax callable."""
  func, weights, buffer = torch._functorch.make_functional_with_buffers(mod)
  states = (weights, buffer)

  @jax.jit
  def jax_func(states, inputs):
    (states, inputs) = tensor.wrap((states, inputs))
    weights, buffer = states
    res = func(weights, buffer, *inputs)
    return tensor.unwrap(res)

  return states, jax_func
