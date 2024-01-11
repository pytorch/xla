import jax
import jax.numpy as jnp

a = jnp.array(1.)

@jax.jit
def f(hx):
  # hx = hx + a
  for i in range(10):
      hx = hx + a
  return hx

hx = jnp.array(0.)
result = f(hx) # jax.lax.scan(f, hx, xs=jnp.arange(10))

print(result)

