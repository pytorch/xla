import jax
import jax.numpy as jnp

a = jnp.array(1.)

# @jax.jit
def f(_, hx):
  hx = hx + a
  return hx

hx = jnp.array(0.)
result = jax.lax.fori_loop(0, 10, f, hx) # jax.lax.scan(f, hx, xs=jnp.arange(10))

print(result)

