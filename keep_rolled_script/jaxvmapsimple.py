import jax
import jax.numpy as jnp

xxx = jnp.arange(9)

@jax.jit
def f(hx):
  hx = hx + 1.
  return hx

vfuc = jax.vmap(f, (0), 0)

print(vfuc(xxx))
