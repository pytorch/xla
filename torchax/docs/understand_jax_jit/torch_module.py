import jax
import jax.numpy as jnp
import torch
from torch.nn.functional import linear

## Goal 1. Illustrate that class attr for torch modules
## dont play nice with jax.jit
## How we solve it.


class Linear(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn(1000, 1000))
    self.bias = torch.nn.Parameter(torch.randn(1000))

  def forward(self, X):
    return linear(X, self.weight, self.bias)


# Running with torch native

print('---- example 1 -----')
m = Linear()
x = torch.randn(2, 1000)
print(m(x))
print(m.forward(x))

with torch.inference_mode():
  # with torch.no_grad():
  print(m.forward(x))

print('---- example 2 -----')

import torchax

env = torchax.default_env()

with env:
  m2 = Linear()
  x = torch.randn(2, 1000)
  print(m2.forward(x))

print('---- example 3 -----')
# where is the jax jit?

# m2 is a callable that takes in Tensor and returns Tensor
#    m2: (Tensor -> Tensor)

# suppose t2j (Tensor -> jax.Array) "unwraps the XLATensor"
# suppose j2t (jax.Array  -> Tensor) "wraps the XLATensor"
from torchax import tensor
import jax


def t2j(torch_tensor: tensor.Tensor) -> jax.Array:
  return torch_tensor._elem


def j2t(jax_array: jax.Array) -> tensor.Tensor:
  return tensor.Tensor(jax_array, env)


# # further notice t2j(j2t(x)) == x; j2t(t2j(x)) == x


def jax_m(X: jax.Array):
  X_jax = j2t(X)
  res = m2(X_jax)
  return t2j(res)


jax_x = jnp.ones((10, 1000))
print(jax_m(jax_x))

## Let f: Tensor -> Tensor
## There is a function g: jax.Array -> jax.Array;
##   g = x |-> j2t (f (t2j(x))). OR,
##   g = j2t . f . t2j (. denotes function composition)
# The correspondence f -> g is an isomorphism too.

jitted_jax_m = jax.jit(jax_m)
print(jitted_jax_m(jax_x))
print(jitted_jax_m.lower(jax_x).as_text())

from torch.utils import _pytree as pytree


def jax_m_functional(states, X):
  states_torch = pytree.tree_map(j2t, states)
  X = j2t(X)
  old_state_dict = m2.state_dict()
  m2.load_state_dict(states_torch, assign=True, strict=False)
  res = m2(X)
  m2.load_state_dict(old_state_dict, assign=True, strict=False)
  return t2j(res)


jax_weights = {
    'weight': m2.weight._elem,
    'bias': m2.bias._elem,
}

jitted_jax_m_functional = jax.jit(jax_m_functional)
print(jitted_jax_m_functional.lower(jax_weights, jax_x).as_text())

# ## interop module

# print('---- exmaple 4 ----')
# import torchax.interop

# def m_functional(states, x):
#   return torch.func.functional_call(m2, states, x)

# with jax.checking_leaks():
#   print(torchax.interop.jax_jit(m_functional)(m2.state_dict(), x))

# # Experiment if time:
# # 1. torch buffer persistence = False
# # 2. torch attr
