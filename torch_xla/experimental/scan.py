"""Module implementing the `scan` higher order operator.

Reference: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html

# High level design

The implementation is factored into two layers: core and autograd. The core
layer focuses on the numerical scan operation without any gradient tracking, and
the autograd layer adds forward and backward support using the scan primitive in
core.

## Core

The `_scan_impl_flat` function implements the core logic of scan on flattened
tensors. It uses XLA's `While` op to iterate over the leading dimension of the
input tensors. The body of the `While` loop calls `fn` and updates the carry and
output tensors.

The `_scan_impl_pytree` function adds PyTree support on top. It flattens the
input PyTrees, calls `_scan_impl_flat` to perform the scan on the flattened
tensors, and then unflattens the results. Because gradients are sometimes
`None`, it also hides any `None`s in PyTrees from `_scan_impl_flat`,
simplifying the latter's implementation.

## Autograd

The `value_and_grad_partitioned` function symbolically traces the user-provided
function `fn` to obtain the forward and backward computation graphs. It then
creates two functions, `forward` and `backward`, that can be used in the
`Scan.forward` and `Scan.backward` methods.

The `scan` operator is implemented as a PyTorch autograd Function, `Scan`.
The `Scan.forward` method scans the forward graph over the inputs.
The `Scan.backward` method scans the backward graph over the gradients and
activations.
"""

import itertools
from typing import Callable, Dict, Sequence, TypeVar, Tuple, List, Optional, overload

import torch
import torch.autograd
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten, tree_iter, PyTree
from functorch.compile import aot_function, make_boxed_func, default_partition  # type: ignore

import torch_xla
import torch_xla.core.xla_builder as xb
from torch_xla.experimental.pytreeify import pytreeify
import torch_xla.debug.profiler as xp

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def scan(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    partition_fn=default_partition,
    # TODO: consider exposing knobs to control the RNG seed used in each `fn` iteration.
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

  Notes:

    `fn` must be AOTAutograd traceable. That requires PyTorch to understand the operations
    within. For example if you invoke a custom kernel inside `fn`, you need to register the
    custom kernel. See https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html.

  Args:
    fn: A Python callable that accepts two PyTrees of tensors: the carry object and the
      slices of `xs` along its leading dimension. It should return two PyTrees: the carry
      object and the slices of the output. The returned carry object will be passed to
      the next invocation of `fn`.

    init: The initial carry object passed to the first invocation of `fn`.
    
    xs: The input PyTree to scan over. If `xs` is a tensor, then `fn` will get slices along
      the leading dimension (`xs[i]`). If `xs` is some other PyTree (e.g. tuple of
      tensor), `fn` will get PyTrees of slices. In that case the leading dimension size
      of the leaves in the PyTree must be the same.
        
    partition_fn: (Optional[Callable]) Since `scan` uses AOTAutograd to trace `fn`, you may
      override what computation happen in the forward and backward passes by specifying
      different partition functions. `default_partition` implies no activation checkpointing.
      You may specify `functorch.compile.min_cut_rematerialization_partition` to use min-cut
      based activation checkpointing. You may also write your own partitioner to insert any
      custom logic such as host offloading of activations.

  Returns:
    (carry, ys): A tuple where `carry` is the last carry object returned by `fn`, and
    `ys` is a PyTree with the same structure as `xs`, but where the leaves are formed
    by stacking the leaf outputs of `fn` respectively. This means if your `fn` returns
    `(carry, (y1, y2))` then this function will return
    `(carry, (torch.stack(all_y1), torch.stack(all_y2)))`.
    
  Example:
  
    >>> # Example of using `scan` to implement `torch.cumsum`.
    >>> import torch_xla.runtime
    >>> import torch
    >>> from torch_xla.experimental.scan import scan
    >>> 
    >>> def fn(carry, x):
    >>>   new_carry = carry + x
    >>>   y = new_carry
    >>>   return new_carry, y
    >>> 
    >>> with torch_xla.runtime.xla_device():
    >>>   init = torch.tensor([0.0, 0.0], requires_grad=True)
    >>>   xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    >>>                     requires_grad=True)
    >>> final_carry, ys = scan(fn, init, xs)
    >>> torch_xla.sync()
    >>> print(final_carry)  # Should be [9.0, 12.0]
    >>> print(ys)  # Should be [[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]]

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

  forward, alias_input, backward = value_and_grad_partitioned(
      fn, init, xs, partition_fn=partition_fn)
  carry, ys = Scan.apply(forward, alias_input, backward, init,
                         xs)  # type: ignore
  return carry, ys


def value_and_grad_partitioned(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    partition_fn=default_partition) -> tuple[Callable, Callable, Callable]:
  """
  Given a user `fn` to be scanned over the leading dimension of the input `xs`
  PyTree and an initial carry object `init`, symbolically traces `fn` and
  returns three functions, `forward`, `alias_input`, and `backward`.
  `forward` and `backward` wrap the forward and backward graphs of `fn` and
  plumbs through intermediate activations, while `alias_input` is a memory
  saving optimization. Specifically, given

    `fn(carry, x) -> (new_carry, y)`

  this function will build and return

    `forward(carry, x) -> (new_carry, (y, partial_activations))`

    `alias_input(stack(partial_activations), xs) -> stack(activations)`

    `backward(grad_new_carry, (grad_y, activations)) -> (grad_carry, grad_x)`

  where `grad_y` is the gradient w.r.t `y`, and `grad_new_carry` is the gradient
  w.r.t. `new_carry`.

  The `partial_activations` returned by `forward` are intermediate activations
  that do not alias any input tensors. You may pass a stack of `partial_activations`
  and the original input `xs` PyTree to `alias_input` to reconstitute the full
  list of `activations`.

  `activations` will always be a flat list of tensors.

  This is similar to the `value_and_grad` transform found in JAX, but additionally
  partitions and returns separate forward/backward passes, so that we may later
  use them in the `autograd.Function` implementation of `Scan`.

  Args:
    fn: (Callable[[Carry, X], tuple[Carry, Y]]) A callable with signature
      `fn(carry, x_t) -> (new_carry, y_t)`, representing the function to be scanned.

    init: (Carry) The initial carry object.

    xs: (X) A PyTree of inputs to be scanned over.

    partition_fn: An optional partitioning function used to partition fn into
      forward and backward graphs.

  Returns:
    A tuple of `(forward, alias_input, backward)`, detailed in the docstring of this function.
  """

  # Make some fake tensors to trace the user function and obtain the
  # forward and backward graphs. Note that the init/carry fake tensor
  # always requires grad. That's because even if the user passed in some
  # `init` that does not require grad, we still want gradients to flow
  # through the `carry` from one iteration of the user function to the
  # next. In summary, the `carry` argument used to trace a user function
  # to get a correct backward pass always requires grad.
  def make_fake_tensor(v: torch.Tensor, requires_grad=True) -> torch.Tensor:
    return torch.empty_like(
        v, dtype=v.dtype, device=v.device, requires_grad=requires_grad)

  fake_carry_pytree = tree_map(make_fake_tensor, init)
  fake_x_pytree = tree_map(
      lambda v: make_fake_tensor(v[0], requires_grad=v.requires_grad), xs)

  # If an output of `fn` aliases the input, `aot_function` will handle that
  # pair of variables with an epilogue inside its generated autograd.Function
  # that we can't access. In other words, the captured graph won't contain
  # those variables. See description in
  # https://github.com/pytorch/pytorch/issues/85036.
  #
  # Because we're abusing AOTAutograd to capture the graph, we need AOTAutograd
  # to handle all variables in the graph as opposed to in the opaque epilogue.
  # This wrapper clones an output if it aliases an input. This hack can go away
  # if scan is instead implemented as a Dynamo compiler pass.
  def fn_no_output_aliasing(*args):
    inputs = set(tree_iter(args))
    return tree_map(lambda v: v.clone() if v in inputs else v, fn(*args))

  with torch.enable_grad():
    fw_compiler, get_fwd = _make_get_graph_compiler()
    bw_compiler, get_bwd = _make_get_graph_compiler()
    fn_compiled = aot_function(
        fn_no_output_aliasing,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn)
    _, unflatten_bwd_out = tree_flatten_none((fake_carry_pytree, fake_x_pytree))
    out = fn_compiled(fake_carry_pytree, fake_x_pytree)
    # How many outputs out of the fwd_graph is actually outputs of `fn`, and not
    # intermediate activations.
    num_out = len(list(tree_iter(out)))
    # Capture the backward.
    out, unflatten_fwd_out = tree_flatten_none(out)
    torch.autograd.backward(out, tree_map(lambda v: torch.ones_like(v), out))

  fwd_graph = get_fwd()
  bwd_graph = get_bwd()

  # Figure out which activations are alises to the inputs. We don't need to
  # pass them through the scan logic unchanged. That would use more memory.
  input_activation_aliases = _find_input_activation_aliases(
      fake_carry_pytree, fake_x_pytree, num_out, fwd_graph)
  aliased_activations = set(input_activation_aliases.values())

  def forward_core(carry, x):
    flat_carry, _ = tree_flatten(carry)
    flat_x, _ = tree_flatten(x)
    with xp.Trace('aot_forward'):
      out = fwd_graph(*flat_carry, *flat_x)
    actual_out, activations = split(out, num_out)
    carry, y = unflatten_fwd_out(actual_out)
    y = (y, activations)
    return carry, y

  def forward(carry, x):
    carry, (y, activations) = forward_core(carry, x)

    # Remove activations that alias to inputs. Those will be added back
    # in `alias_input`.
    partial_activations = tuple(
        v for i, v in enumerate(activations) if i not in aliased_activations)

    y = (y, partial_activations)
    return carry, y

  def alias_input(partial_activations, xs):
    """
    Add back activations that are aliases to input tensors.

    In principle, we could have `forward` return all the intermediate activations,
    including those that are aliases to an input tensor. However, those inputs will
    then be duplicated as part of the output of a `scan` call, because we want to
    save all activations during the forward pass of a `scan`. The XLA compiler can't
    optimize away this duplication likely because they're behind a DynamicSlice +
    DynamicUpdateSlice, so we end up doubling the memory usage from those inputs.

    To reduce memory usage, we can have `forward` return the activations that
    don't alias to inputs, called `partial_activations`. The autograd implementation
    of `scan` will call `alias_input` to add back activations that are aliases
    of input tensors outside of a scan, turning the partial activations back to
    full activations.
    """
    activations = list(partial_activations)
    aliased_inputs = [
        v for i, v in enumerate(tree_iter(xs)) if i in input_activation_aliases
    ]
    for (i, activation_idx) in enumerate(input_activation_aliases.values()):
      activations.insert(activation_idx, aliased_inputs[i])
    return tuple(activations)

  def backward(carry, x):
    grad_new_carry, _ = tree_flatten(carry)
    (grad_y, activations) = x
    grad_y, _ = tree_flatten_none(grad_y)
    with xp.Trace('aot_backward'):
      out = bwd_graph(*activations, *grad_new_carry, *grad_y)
    grad_carry, grad_x = unflatten_bwd_out(out)
    return grad_carry, grad_x

  return forward, alias_input, backward


def _find_input_activation_aliases(fake_carry_pytree, fake_x_pytree, num_out,
                                   fwd_graph):
  """
  Find which activations are aliases to input tensors.

  Returns:

    A mapping from index into the flatttened
    input pytree to the index into the list of intermediate activations.

  """
  flat_carry, _ = tree_flatten(fake_carry_pytree)
  flat_x, _ = tree_flatten(fake_x_pytree)
  _actual_out, activations = split(fwd_graph(*flat_carry, *flat_x), num_out)
  input_id_to_index = {
      v: i for i, v in enumerate(id(v) for v in tree_iter(flat_x))
  }
  input_activation_aliases = {}
  for idx, i in enumerate(id(v) for v in activations):
    if i in input_id_to_index:
      input_activation_aliases[input_id_to_index[i]] = idx
  return input_activation_aliases


def _make_get_graph_compiler():
  """
  Creates a compiler that records the graph, and a getter
  function to retrieve them.
  """
  graph: List[Optional[torch.fx.GraphModule]] = [None]

  def forward_comp(fx_module: torch.fx.GraphModule, _):
    assert graph[0] is None
    graph[0] = fx_module
    return make_boxed_func(fx_module)

  def get_graph():
    g = graph[0]
    assert g is not None
    return g

  return forward_comp, get_graph


@pytreeify
class Scan(torch.autograd.Function):

  @staticmethod
  def forward(ctx, forward, alias_input, backward, init, xs):
    ctx._backward = backward
    with torch.no_grad():
      carry, ys = _scan_impl_pytree(forward, init, xs)
      ys, partial_activations = ys
    activations = alias_input(partial_activations, xs)
    ctx.save_for_backward(*activations)
    return carry, ys

  @staticmethod
  def backward(ctx, grad_carry, grad_ys):  # type: ignore
    activations = ctx.saved_tensors
    backward = ctx._backward
    with torch.no_grad():
      # Reverse loop to propagate gradients from last iteration to first.
      grad_init, grad_xs = _scan_impl_pytree(
          backward, grad_carry, (grad_ys, activations), reverse=True)
    return None, None, None, grad_init, grad_xs


def _scan_impl_pytree(fn, init, xs, reverse: bool = False):
  """Forward logic of scan without gradient tracking. `fn` operates on
  PyTrees. `init` and `xs` are also PyTrees.
  
  See the `Scan` class which implements an autograd `Function` and builds
  autograd support on top of `_scan_impl`.
  """
  flat_init, unflatten_carry = tree_flatten_none(init)
  flat_xs, unflatten_xs = tree_flatten_none(xs)
  unflatten_y: Callable[..., PyTree] = lambda _: ()  # Set by `flat_fn`.

  def flat_fn(
      carry: Sequence[torch.Tensor], x: Sequence[torch.Tensor]
  ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    nonlocal unflatten_y
    carry_pytree = unflatten_carry(carry)
    x_pytree = unflatten_xs(x)
    carry_pytree, y_pytree = fn(carry_pytree, x_pytree)
    flat_carry, _ = tree_flatten_none(carry_pytree)
    flat_y, unflatten_y = tree_flatten_none(y_pytree)
    return flat_carry, flat_y

  flat_carry, flat_y = _scan_impl_flat(
      flat_fn, flat_init, flat_xs, reverse=reverse)
  return unflatten_carry(flat_carry), unflatten_y(flat_y)


def tree_flatten_none(pytree: PyTree):
  """
  Flattens input `pytree`, and filters out any `None` leaf PyTree nodes.
  Returns the flattened list, and an unflatten function and also adds back
  the removed `None`s in their correct location.
  """
  flat, spec = tree_flatten(pytree)
  flat, add_none = _remove_none(flat)

  def unflatten(flat):
    flat = add_none(flat)
    return tree_unflatten(flat, spec)

  return flat, unflatten


def _remove_none(s: Sequence[Optional[torch.Tensor]]):
  """
  Filters out `None` values from `s`. Returns the filtered sequence,
  and another function that will add back the `None` values when given a
  sequence of the same structure.
  """
  filtered = [v for v in s if v is not None]
  none_mask = [v is None for v in s]

  def add_back_nones(s_filtered):
    res = []
    idx_filtered = 0
    for is_none in none_mask:
      if is_none:
        res.append(None)
      else:
        res.append(s_filtered[idx_filtered])
        idx_filtered += 1
    return res

  return filtered, add_back_nones


def dynamic_update_slice(ys: xb.Op, y: xb.Op, idx: xb.Op) -> xb.Op:
  # See https://openxla.org/xla/operation_semantics#dynamicupdateslice.
  y = y.broadcast([1])
  indices = [idx]
  for _ in range(ys.shape().rank - 1):
    indices.append(idx.zeros_like())
  return ys.dynamic_update_slice(y, indices)


def dynamic_slice(xs: xb.Op, idx: xb.Op) -> xb.Op:
  indices = [idx]
  for _ in range(xs.shape().rank - 1):
    indices.append(idx.zeros_like())
  slice_shape = list(xs.shape().sizes)
  slice_shape[0] = 1
  sliced = xs.dynamic_slice(indices, slice_shape)
  shape = list(xs.shape().sizes)
  shape = shape[1:]
  return sliced.reshape(shape)


class Builder:

  def __init__(self, name: str):
    self._builder = xb.create_builder(name)
    self._params = []
    self._param_tensors = []

  def add_param(self, val: torch.Tensor):
    idx = len(self._params)
    param = xb.mkparam(self._builder, idx, xb.tensor_shape(val))
    self._params.append(param)
    self._param_tensors.append(val)
    return idx

  def params(self) -> Tuple[xb.Op, ...]:
    return tuple(self._params)

  def param_tensors(self) -> Tuple[torch.Tensor, ...]:
    return tuple(self._param_tensors)

  def num_params(self) -> int:
    return len(self._params)


def _scan_impl_flat(fn,
                    init: Sequence[torch.Tensor],
                    xs: Sequence[torch.Tensor],
                    reverse: bool = False):
  """Forward logic of scan without gradient tracking. `fn` operates on
  two flat list of tensors. `init` and `xs` are also flat lists of tensors. None
  of the tensors will be `None`.
  
  See the `Scan` class which implements an autograd `Function` and builds
  autograd support on top of `_scan_impl`.
  
  ## Handling of random numbers

  When `fn` generates random numbers (e.g. it uses a dropout layer), we need to
  ensure that each iteration of `fn` within the scan yields different random
  numbers, despite running the same HLO operations. JAX requires the user to
  explicitly fork the RNG state and pass it to `fn`. In PyTorch, the RNG state
  is an implicit global variable. Therefore, we take a slightly different
  approach:

  - Identify usage of RNG state via `_get_tensors_xla_device_data_node`.
  - Create N different copies of the RNG state contained in a tensor.
  - While building the `While` op body, index into the RNG state tensor at the
    current iteration and provide that seed value to `fn`.
  
  ## Handling of HLO parameters
  
  Let's say the user writes a `fn` like this:
  
    def fn(carry, x):
      foo = torch.zeros(8)
      return carry, x + foo
  
  `fn` will lower into an HLO computation like this:

    HloModule Fn, entry_computation_layout={
      (f32[8], f32[8], f32[8]) -> (f32[8], f32[8])
    }
  
  The HLO computation takes three parameters while `fn` takes two arguments.
  That's because IR lowering does not distinguish if a leaf data tensor comes from
  a function argument or from within the function. All data tensors are lowered
  into HLO parameters. We'll call them "hoisted variables" or `hoisted_vars`, since
  instead of baking the value of those tensors as literals in the HLO graph,
  they are turned into additional parameters of the computation.
  """
  carry_len = len(init)
  xs_len = len(xs)

  # Abstractly trace and lower `fn`.
  # Later we will include `fn_computation` within the while loop body.
  def make_fake_tensor(v: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        v.size(), dtype=v.dtype).to(device).requires_grad_(v.requires_grad)

  device = torch_xla.device()
  fake_carry = tree_map(make_fake_tensor, init)
  fake_x = tree_map(lambda v: make_fake_tensor(v[0]), xs)
  fake_output_carry, fake_output_y = fn(fake_carry, fake_x)

  y_len = len(fake_output_y)
  fn_outputs = fake_output_carry + fake_output_y

  fn_ctx = torch_xla._XLAC.lowering.LoweringContext("FnComputation")
  fn_ctx.set_name_string("fn_ctx")
  fn_ctx.build(list(fn_outputs))
  fn_hlo = fn_ctx.hlo()
  fn_computation = xb.computation_from_module_proto("fn_computation", fn_hlo)

  # Figure out the shape of `ys` from the abstract tracing.
  fn_carry_out, fn_y_out = split(fn_outputs, carry_len)
  assert carry_len + y_len == len(fn_outputs)
  fn_carry_shapes = [v.shape for v in fn_carry_out]
  for fn_carry_shape, init_leaf in zip(fn_carry_shapes, init):
    assert fn_carry_shape == init_leaf.shape, f"`fn` must keep the `carry` shape unchanged. \
      Got {fn_carry_shape} but expected {init_leaf.shape}"

  builder = Builder('scan')
  num_iters = next(iter(tree_iter(xs))).size(0)
  ys = [
      torch.zeros((num_iters, *v.shape), device=device, dtype=v.dtype)
      for v in fn_y_out
  ]
  # Start the `curr_iter` loop variable at zero.
  zero = torch.tensor(0, device=device)
  builder.add_param(zero)

  # We are building a bigger XLA computation (the while loop) that calls
  # a smaller computation (`fn_computation`). This is a mapping from
  # `fn_computation` param ID to While computation param ID.
  fn_param_id_to_while_param_id: Dict[int, int] = {}

  # Add carry and x.
  for real, fake in ((init, fake_carry), (xs, fake_x)):
    for val, fake_val in zip(real, fake):
      idx = builder.add_param(val)
      param_id = fn_ctx.tensor_parameter_id(fake_val)
      if param_id != -1:
        fn_param_id_to_while_param_id[param_id] = idx

  # Add the output as a param since our While computation consumes it, updates
  # one slice, and returns the updated ys in each iteration.
  for val in ys:
    builder.add_param(val)

  # Detect hoisted variables.
  hoisted_vars: Dict[
      int, torch.Tensor] = fn_ctx.device_parameter_id_tensor_mapping()
  for v in itertools.chain(fake_carry, fake_x):
    param_id = fn_ctx.tensor_parameter_id(v)
    if param_id != -1:
      del hoisted_vars[param_id]

  # Detect RNG seed usage within the scanned function within hoisted variables.
  ids, i_values = torch_xla._XLAC._get_tensors_xla_device_data_node(fn_outputs)
  seed_info_id = torch_xla._XLAC._get_seed_info_id()
  seed_parameter_id = None
  if seed_info_id in ids:
    seed_idx = ids.index(seed_info_id)
    seed_parameter_id = fn_ctx.tensor_parameter_id(i_values[seed_idx])
    assert seed_parameter_id != -1, "`fn` uses random seed, but random seed is not \
      a parameter to the traced HLO graph"

    # Replace the single seed value with a tensor of seeds, one per iteration.
    seed_tensor = hoisted_vars[seed_parameter_id]
    assert seed_tensor.dtype == torch.int64
    hoisted_vars[seed_parameter_id] = torch.randint(
        0, 2**62, (num_iters,), dtype=torch.int64, device=torch_xla.device())

  # Add hoisted variables as While computation params as well,
  # including the potentially updated seed tensor.
  for param_id, tensor in hoisted_vars.items():
    idx = builder.add_param(tensor.to(torch_xla.device()))
    fn_param_id_to_while_param_id[param_id] = idx

  # Since we are threading five objects through the body_fn:
  #
  # - curr_iter: the current loop iteration
  # - carry: the scan state
  # - xs: the flattened input pytree
  # - ys: the flattened output of fn
  # - hoisted_vars: tensors not provided as arguments to fn but still used by fn.
  #
  # We need to concatenate all into one big list prior to entering `body_fn` and
  # `cond_fn`, and split them back which is easier to work with after that. This
  # pair of `pack`, `unpack` functions is for that purpose.
  T = TypeVar('T')

  def pack(curr_iter: T, carry: Sequence[T], xs: Sequence[T], ys: Sequence[T],
           hoisted_vars: Sequence[T]) -> Tuple[T, ...]:
    return tuple(itertools.chain((curr_iter,), carry, xs, ys, hoisted_vars))

  def unpack(seq: Sequence[T]) -> Tuple[T, List[T], List[T], List[T], List[T]]:
    curr_iter, carry, xs, ys, hoisted_vars = split(
        list(seq), 1, carry_len, xs_len, y_len)
    curr_iter = curr_iter[0]
    return curr_iter, carry, xs, ys, hoisted_vars

  def replace_rng_seed(curr_iter: xb.Op, *while_params: xb.Op):
    """Slices the pre-generated seed tensor for the current iteration."""
    if seed_parameter_id is None:
      return while_params
    idx = fn_param_id_to_while_param_id[seed_parameter_id]
    replaced = list(while_params)
    replaced[idx] = dynamic_slice(replaced[idx], curr_iter)
    return replaced

  def call_fn_computation(*while_params: xb.Op) -> xb.Op:
    # We need to order the tensors in increasing parameter ID order when
    # passing them to `xb.Op.call`.
    fn_inputs = [
        while_params[fn_param_id_to_while_param_id[i]]
        for i in range(len(fn_param_id_to_while_param_id))
    ]
    return xb.Op.call(fn_computation, fn_inputs)

  def cond_fn(curr_iter: xb.Op, *rest):
    return curr_iter < xb.Op.scalar(
        curr_iter.builder(), num_iters, dtype=xb.Type.S64)

  def body_fn(*while_params: xb.Op):
    curr_iter, carry, xs, ys, hoisted_vars = unpack(while_params)
    if reverse:
      max_iter = xb.Op.scalar(
          curr_iter.builder(), num_iters - 1, dtype=xb.Type.S64)
      idx = max_iter - curr_iter
    else:
      idx = curr_iter
    x = [dynamic_slice(v, idx) for v in xs]
    result = call_fn_computation(
        *replace_rng_seed(idx, curr_iter, *carry, *x, *ys, *hoisted_vars))
    for i in range(carry_len):
      carry[i] = result.get_tuple_element(i)
    for i in range(y_len):
      y = result.get_tuple_element(i + carry_len)
      ys[i] = dynamic_update_slice(ys[i], y, idx)
    one = xb.Op.scalar(curr_iter.builder(), 1, dtype=xb.Type.S64)
    return pack(curr_iter + one, carry, xs, ys, hoisted_vars)

  res = xb.Op.mkwhile(builder.params(), cond_fn, body_fn)
  computation = res.build('scan')
  outputs = torch_xla._XLAC._xla_user_computation('xla::scan',
                                                  builder.param_tensors(),
                                                  computation)
  _curr_iter, carry, xs, ys, _hoisted_vars = unpack(outputs)
  return carry, ys


U = TypeVar('U')


@overload
def split(seq: List[U], *part_lengths: int) -> Tuple[List[U], ...]:
  ...


@overload
def split(seq: Tuple[U, ...], *part_lengths: int) -> Tuple[Tuple[U, ...], ...]:
  ...


def split(seq: Sequence[U], *part_lengths: int) -> Tuple[Sequence[U], ...]:
  """Splits a sequence into subsequences with given lengths.

  Args:
    seq: The sequence (list or tuple) to split.
    *part_lengths: The lengths of the subsequences, except the last subsequence.

  Example:

    a, b, c = split((1, 2, 3, 4, 5), 2, 2)
    # a == (1, 2), b == (3, 4), c == (5, )

  Returns:
    A tuple of subsequences (lists or tuples).
  """
  parts = []
  start = 0
  for length in part_lengths:
    parts.append(seq[start:start + length])
    start += length
  parts.append(seq[start:])
  return tuple(parts)
