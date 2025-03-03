import torch
import torch_xla.core.xla_model as xm

dev = xm.xla_device()

a = torch.ones((3,3), device=dev)

import torch_xla.core.xla_builder as xb 

import jax.numpy as jnp
def f(a, b):
  return a + jnp.sin(b)

b = xb.call_jax(f, (a, a), {}, 'hame')
print(b)