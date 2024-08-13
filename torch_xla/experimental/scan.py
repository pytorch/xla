"""Module implementing the `scan` higher order operator.

Reference: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html

"""

from typing import Callable, TypeVar

import torch
from torch.utils._pytree import tree_map, tree_iter

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def scan(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
) -> tuple[Carry, Y]:
  """Apply a function over leading dimension of tensors while carrying along state.
  
  This is similar to the JAX `jax.lax.scan` function found in [1].
  
  You may use it to loop over the leading dimension of tensors efficiently. If `xs`
  is a single tensor, this function is roughly equal to the following Python code:

    def scan(fn, init, xs):
      ys = []
      carry = init
      for i in len(range(xs.size(0))):
        carry, y = fn(carry, xs[i])
        ys.append(y)
      return carry, torch.stack(ys, dim=0)
  
  In the general case, `Carry`, `X`, and `Y` can be arbitrary PyTrees. This function
  will iterate through the leading dimension of every leaf element of `xs` simultaneously,
  and pass a slice of those elements to `fn` as another PyTree. This means you may
  scan over multiple tensors and produce multiple output tensors at once.
  
  Args:

    fn: a Python callable that accepts two PyTrees of tensors: the carry object and the
        slices of `xs` along its leading dimension. It should return two PyTrees: the carry
        object and the slices of the output. The returned carry object will be passed to
        the next invocation of `fn`.

    init: the initial carry object passed to the first invocation of `fn`.
    
    xs: the input PyTree to scan over. If `xs` is a tensor, then `fn` will get slices along
        the leading dimension (`xs[i]`). If `xs` is some other PyTree (e.g. tuple of
        tensor), `fn` will get PyTrees of slices. In that case the leading dimension size
        of the leaves in the PyTree must be the same.

  Returns:

    (carry, ys): A tuple where `carry` is the last carry object returned by `fn`, and
    `ys` is a PyTree with the same structure as `xs`, but where the leaves are formed
    by stacking the leaf outputs of `fn` respectively. This means if your `fn` returns
    `(carry, (y1, y2))` then this function will return
    `(carry, (torch.stack(all_y1), torch.stack(all_y2)))`.

  [1]: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
  """

  # Ensure that `fn` is callable.
  if not callable(fn):
    raise ValueError(f"`fn` {fn} must be callable.")

  # Ensure that the leaves have the same length.
  xs_length = None
  for leaf in tree_iter(xs):
    leaf_len = len(leaf)
    if xs_length is None:
      xs_length = leaf_len
    if xs_length != leaf_len:
      raise ValueError(
          f"The leaves of the `xs` input PyTree must have the same leading dimension size. \
                       Got {xs_length} and {leaf_len}")

  if xs_length is None:
    raise ValueError(f"`xs` {xs} is an empty PyTree.")

  carry = init
  ys = []

  for i in range(xs_length):
    carry, y = fn(carry, tree_map(lambda x: x[i], xs))
    ys.append(y)

  # Combine the list of PyTrees into one PyTree, where the leaves are
  # stacked into a new major axis.
  ys = tree_map(lambda *x: torch.stack(x), *ys)
  return carry, ys
