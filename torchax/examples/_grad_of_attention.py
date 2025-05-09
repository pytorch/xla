import jax.numpy as jnp
import jax
from jax.experimental.pallas.ops.tpu import flash_attention

import torchax
from jax.experimental import mesh_utils
from torchax.ops.jtorch import _tpu_flash_attention

env = torchax.default_env()
jax.config.update('jax_enable_x64', False)
env._mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((4,)),
    axis_names=("fsdp",),
)
env.use_flash_attention = True

from torch.nn import functional as F


def attn(q, k, v):
  q, k, v = env.j2t_iso((q, k, v))
  with env:
    x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
  x = env.t2j_iso(x)
  return jnp.sum(x)


import torch


class M(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.a = torch.nn.Linear(10, 10)

  def forward(self, x):
    return self.a(x)


m = M()
from torchax.interop import JittableModule

mjit = JittableModule(m)

from torch.nn.utils import stateless


def f(weights, x):
  res = mjit.functional_call('forward', weights, {}, (x,))
  return torch.sum(res)


def crossent(x, y):
  x, y = env.j2t_iso((x, y))
  res = torch.func.functional_call(m, x, (y,))
  return env.t2j_iso(res)


graded = jax.value_and_grad(attn)

shape = (4, 32, 128, 32)
q = jnp.ones(shape, dtype='bfloat16')
v = jnp.ones(shape, dtype='bfloat16')
k = jnp.ones(shape, dtype='bfloat16')

env = torchax.default_env()
weights = env.t2j_iso(env.to_xla(mjit.params))

from torchax.interop import jax_view

#print(jax.jit(graded).lower(q, v, k).as_text())
print(
    jax.jit(jax.grad(jax_view(f))).lower(weights,
                                         jax.ShapeDtypeStruct(
                                             (10,), 'float32')).as_text())
