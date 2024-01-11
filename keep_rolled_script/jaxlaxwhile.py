import jax
import jax.numpy as jnp

a = jnp.array(1.)

# @jax.jit
def f(hx):
  i = 0
  while i < 10:
    hx = hx + a
    i = i + 1
  return hx

hx = jnp.array(0.)
result = f(hx)

print(result)

