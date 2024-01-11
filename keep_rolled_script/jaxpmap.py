import jax
import jax.numpy as jnp

a = jnp.ones(1)

print("a", a)

@jax.jit
def f(hx):
  hx = hx + a
  return hx

# use 4 here due to local has 4 xla devices
result = jax.pmap(lambda x: x + a)(jnp.arange(4))

print(result)
