import jax
import jax.numpy as jnp

a = jnp.array(1.)

# @jax.jit
def f(hx):
  hx = hx + a
  return hx

hx = jnp.array(0.)
result = jax.lax.while_loop(lambda hx: hx < 10, f, hx) # jax.lax.scan(f, hx, xs=jnp.arange(10))

print(result)

