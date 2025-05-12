## Please read: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html

## Goal 1. Jax jit what it is
## Goal 2. Illustrate that changing shapes == recompile
## Goal 3. Illustrate that closures == inlined constant in graph (undesirable)

import random
import time
import jax.numpy as jnp
import numpy as np
from jax import jit
import jax


def norm(X):
  for i in range(10):
    X = X @ X
    X = X - X.mean(0)
  return X / X.std(0)


norm_compiled = jit(norm)

np.random.seed(1701)

# print('---- example 1 -----')
# for i in range(5):
#   X = jnp.array(np.random.rand(1000, 1000))
#   start = time.perf_counter()
#   norm(X).block_until_ready()
#   end = time.perf_counter()
#   print(f'iteration {i}: norm: {end - start}')
#   start = time.perf_counter()
#   norm_compiled(X).block_until_ready()
#   end = time.perf_counter()
#   print(f'iteration {i}: norm_compiled: {end - start}')

#jax.config.update("jax_explain_cache_misses", True)
#print('---- example 3 -----')
#print('---- example 2 -----')
# for i in range(5):
#   shape = random.randint(1000, 2000)
#   print('shape is ', shape)
#   X = jnp.array(np.random.rand(shape, shape))
#   start = time.perf_counter()
#   norm(X).block_until_ready()
#   end = time.perf_counter()
#   print(f'iteration {i}: norm: {end - start}')
#   start = time.perf_counter()
#   norm_compiled(X).block_until_ready()
#   end = time.perf_counter()
#   print(f'iteration {i}: norm_compiled: {end - start}')

#Example 4: print out the graph
print('--- example 4 ---')

X = jnp.array(np.random.rand(1000, 1000))

#print(norm_compiled.lower(X).as_text())
#print(norm_compiled.lower(jax.ShapeDtypeStruct((1000, 1000), jnp.float32.dtype)).as_text())

# Example 5: What happen to closures
print('--- example 5 ---')


def addx(y):
  #print('y is ', y)
  #print('X is ', X)
  # import pdb; pdb.set_trace()
  #jax.debug.print(...)
  #jax.debug.breakpoint()
  return y + X


addx_jitted = jax.jit(addx).lower(
    jax.ShapeDtypeStruct((1000, 1000), jnp.float32.dtype))
#print(addx_jitted.as_text())
#print(addx_jitted.compile()(X))

# ASIDE: pdb; print;
# https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html

# Example 6: What happens with class attr


class Model:

  def __init__(self):
    self.weight = jnp.array(np.random.randn(1000, 1000))
    self.bias = jnp.array(np.random.randn(1000))

  def __call__(self, X):
    return X @ self.weight + self.bias


m = Model()
print('Not jitted', m(X))
print(jax.jit(m).lower(X).as_text())


def model_pure(weight, bias, X):
  m.weight = weight
  m.bias = bias
  res = m(X)
  return res
