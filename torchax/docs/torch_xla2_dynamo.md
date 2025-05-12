# Dynamo backend for torchxla2

## Goal

Have a dynamo backend backend by torchax.

The users should be able to do the following:

```python
m = model ...
m_compiled = torch.compile(m, backend='torchax_compile')  # backend name TBD
result = m_compiled(*inputs)
```

The above should run on TPU will low overhead.

## Challenge

Usually the challenge of a dynamo backend is the compiler that
transforms a fx graph with torch (or Aten) ops to the compiled executable.
However, in our case, that piece is solved.

For every `call_function` node; we lookup the corresponding implementation of
said ATen op in a dictionary for it's corresponding implementation in Jax,
and we just call it.

This is illustrated here: https://github.com/pytorch/xla/blob/master/experimental/torchax/torchax/export.py#L23

Now, the challenge is for dynamo to be able to 1. produce the graph; and 2. n
not incur any data copies in this process.


Consider this following pseudocode:

```python
class Tensor:
  _data: jax.Array
  def __torch_dispatch__(...):
      # do stuff with _data, get new data
      return Tensor(new_data)

def dynamo_backend(fx, sample):
  compiled = compile fx into graph that manipulate jax.Array.
  def returned_callable(inputs):
    datas = [i._data for i in inputs]
    res = compiled(*datas)
    return TensorSubclass(res)
  return returned_callable

model = torch.compile(model, backend = dynamo_backend)
inputs = a list of TensorSubclass or a list of torch.Tensor?
model(*inputs)
```

What would be the type of inputs?
If inputs are of type `TensorSubclass`, then dynamo
will attempt to trace through the `__torch_dispatch__` method,
and throws error because it doesn't know what is `_data` and the
operations on it.

If `inputs` is of type `torch.Tensor`, then it works: dynamo
calls the backend, the backend can produce correct result.
But, `inputs` need to be converted to `TensorSubclass` first inside of
the backend; which usually means a data copy. This happens everytime
the compiled backend is executed, therefore not desirable.

## The Desired behavior

When *tracing* dynamo treats TensorSubclass as if it is a regular tensor
without dispatch override; and when executing the compiled callable,
TensorSubclass is passed in as-is. We know that dynamo can do this with
some tensor subclass, namely `FakeTensor`.


Let's list out the possible ways we could accomplish this behavior.


# Option 1. Have the jax.Array object hold in C++

Roughly we would have a `Tensor` subclass in C++, this is very
similar to the `LazyTensor` subclass that is the current `XLATensor`.
This tensor can hold it's own states in C++. In our case, that would
be a `PyObject*` that happens to point to either `jnp.ndarray` or
jax's `Traced<ShapedArray>` during jax.jit. We might further result the
`XLA` dispatch key to route the operators to the jax implementation,
emulating what `__torch_dispatch__` does.

This way, eager mode will continue to work, and dynamo would work
because the Python class is still `torch.Tensor` (not a subclass), and
there are no Python logic in dispatching so dynamo cannot trace through.

## Pros:
* Very clear that this will work.
* Recommended by ezyang

## Cons:
Now need to deal with C++ builds. In particular, `torch` becomes a source
dependency instead of a pip dependency; meaning, again we need to start
building torch first then build torchax. This might be mitigated if
that subclass can be upstreamed.


# Option 2. Modify dynamo to do the desired behavior

We have one instance where a `torch.Tensor` dispatch subclass
just works with dynamo, without dynamo make a fuss when it traces
`__torch_dispatch__`. This is `FakeTensor`. (https://github.com/pytorch/pytorch/pull/100017/files)

The idea is to make dynamo trace as-if the inputs are `FakeTensor` and
not `XLATensor`. and only after the creation of fx graph and backend, dynamo
calls the compiled callable with `XLATensor`.

Pros:
* Likely pure python changes.

Cons:
* We also need to design a mechanism to represent tensor subclasses that
  is desirable for dynamo to trace through, and those is not.
* Likely significant amount of work.


# Option 3. Register All the ops as custom_ops

So currently dynamo traces `__torch_dispatch__`, and we don't like that
because it will find the operations on Jax arrays, and doesn't understand those.

What if we make dynamo **able** to understand what is inside?
The [Black box python functions](https://docs.google.com/document/d/1ZuCVyMfibExwvtzhd9cfMWk5zXT3Dhy1b3kuvAIkBoU/edit#heading=h.56tggsazyrkh) doc
points the possibility of registering things that we don't want dynamo
to go into as a custom op. So we could, theoretically do the following:

1. Register the jax impl of an Aten op as a custom op.
   i.e. register `jaten.add` for `aten.add`.
2. For meta kernels, just call the meta kernel of `aten.add`.
3. In `__torch_dispatch__`, we forward the call from `aten.add` to `jaten.add`.

When dynamo attempts to go inside of `__torch_dispatch__`, it will find
`jaten.add`. Then it will record that in the `fx.Graph`.

Our backend will see the same ops but in a different namespace (`jaten`).
That is fine as long as we know how to look up its implementation.

Note: we probably also need to hook up gradients of custom ops via. `autograph.Function`.


Pros / Cons:
Haven't tried, don't know if it gonna work or not.






# Appendix, Failed attempts:

## Attempt 1: move dispatch to a mode (i.e. subclass have no dispatch override)

```python
class Subclass(torch.Tensor):

  @staticmethod
  def __new__(cls, elem):
    dtype = tensor.j2t_dtype(elem.dtype)
    shape = list(elem.shape)
    for i, s in enumerate(shape):
      if not isinstance(s, int):
        shape[i] = 1
    if dtype is None:
      dtype = torch.float32

    self = torch.Tensor._make_wrapper_subclass(
        cls,
        shape,
        dtype=dtype,
        device='meta',
        requires_grad=False,
    )
    self._meta = torch.empty(
        shape, dtype=dtype, device='meta', requires_grad=False
    )
    self._elem = elem
    return self

  def __init__(self, elem: jax.Array):
    super().__init__()
    self._elem = elem

  def __str__(self):
    return "Subclass({} {})".format(str(type(self._elem)), str(self._elem))

```

This fails with an error saying that exhausted subclasses and all the `__torch_dispatch__` returned `NotImplemented`.

