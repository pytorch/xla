import jax
import jax.numpy as jnp

# hx = jnp.array(1.)

# @jax.jit
# def f(aa):
#   hx = hx + a
#   return hx

aas = jnp.ones(5) # jnp.array(0.)
# result = jax.lax.map(f, aas) # jax.lax.scan(f, hx, xs=jnp.arange(10))

# print(result)

class Counter:
  """A simple counter."""
  def __init__(self):
    self.n = 0
  def count(self) -> int:
    """Increments the counter and returns the new value."""
    import pdb; pdb.set_trace()
    self.n += 1
    # print("value", value)
    print(self.n)
    return self.n
  def reset(self):
    """Resets the counter to zero."""
    self.n = 0
  def showvalue(self):
    return self.n
    # print(self.n)

counter = Counter()

def f(intvalue):
  print("th", intvalue)
  counter.count()
  return counter.showvalue()

result = jax.lax.map(f, aas)
print(result)
# counter.showvalue()

# for _ in range(10):
#   print(counter.count())
