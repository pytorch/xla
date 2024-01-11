import jax
import jax.numpy as jnp

a = jnp.array(1.)

@jax.jit
def f(hx, _):
  hx = hx + a
  return hx, None

hx = jnp.array(0.)
result = jax.lax.scan(f, hx, xs=jnp.arange(10))

print(result)

