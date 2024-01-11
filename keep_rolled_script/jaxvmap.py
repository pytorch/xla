import jax
import jax.numpy as jnp

A, B, C, D, E = 1, 2, 3, 4, 5
x = jnp.ones((E, A, B)) # 5, 1, 2
y = jnp.ones((B, E, C)) # 2, 5, 3

tree = (x, y)

def foo(tree_arg):
  x, y = tree_arg
  return jnp.dot(x, y)

vfoo = jax.vmap(foo, in_axes=((0, 1),))
print(vfoo(tree).shape)
print(vfoo(tree))
