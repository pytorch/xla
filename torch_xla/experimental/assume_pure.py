from copy import copy
from functools import wraps

import torch
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
import torch_xla
from torch_xla._internal.jax_workarounds import requires_jax
import torch_xla.core.xla_builder as xb

_XLA_COMPUTATION_CACHE = {}


@requires_jax
def assume_pure(fn):
  """Decorates a pure PyTorch/XLA function to skip expensive re-tracing.

  Returns a new function that will only be traced once for each unique
  input tensor shapes or non-tensor input argument values. This is useful
  for removing Lazy Tensor tracing overhead.

  The decorated function must be pure (i.e. no side-effects, behavior
  only depends on inputs).

  Limitations:
  - The decorated function can only use upstream PyTorch operators e.g.
    `torch.einsum`, `torch.nn.functional.layer_norm`, and a few PyTorch/XLA operators:
    * `torch_xla.experimental.assume_pure` (recursive `assume_pure`)
    * `torch_xla.distributed.spmd.mark_sharding`

  - Other custom PyTorch/XLA operations such as `flash_attention` are not
    supported. This limitation may be lifted in the future.
  """
  from torchax.interop import jax_view
  return j2t_autograd(jax_view(fn))


@requires_jax
def j2t_autograd(fn):
  """Given a JAX function, returns a PyTorch autograd function implemented with `jax.vjp(fn)`.

  It wraps `fn` with `jax.vjp` to compute both the output and residuals (intermediate
  activations). The wrapped function is then run via `call_jax` and integrated into
  the PyTorch autograd framework by saving the residuals into the context object.
  """
  import torchax.interop
  return torchax.interop.j2t_autograd(
      fn, call_jax=lambda fn, *args: xb.call_jax(fn, args))


def make_function_tensor_inputs(fn, flat_inputs, input_tree_spec):
  # Create a wrapper function that takes only tensor inputs.
  @wraps(fn)
  def wrapper(*tensor_args):
    # Go from a list of tensor args to the full list of flattened arguments,
    # by referencing the original flattened inputs.
    new_flattened = copy(flat_inputs)
    tensor_args_iter = iter(tensor_args)
    for i in range(len(flat_inputs)):
      if isinstance(flat_inputs[i], torch.Tensor):
        new_flattened[i] = next(tensor_args_iter)
    args, kwargs = tree_unflatten(new_flattened, input_tree_spec)
    out = fn(*args, **kwargs)
    return out

  return wrapper


def pick_tensor_inputs(inputs):
  assert isinstance(inputs, (tuple, list))
  return tuple(i for i in inputs if isinstance(i, torch.Tensor))


def make_fake_inputs(input):
  """Creates a fake input for the given input torch tensor. If the input
  is not a tensor, it returns the input as is.
  """
  if isinstance(input, torch.Tensor):
    t = xb.create_placeholder_tensor(input.shape, input.dtype)
    return t.requires_grad_(input.requires_grad)
  return input


def assume_pure_torch(func=None, use_cache=False):
  """Decorator to mark a function as pure for PyTorch/XLA.
  This decorator builds an XLA computation from the function and caches it.
  The decorated function must be pure (i.e. no side-effects, behavior
  only depends on inputs). 
  Args:
    func: The function to be decorated.
    use_cache: If True, caches the XLA computation for the function with
      the same name as the function. It is the user's responsibility to ensure
      that the function is called with the same input shapes and types each time
      when using this.
  NOTE: This decorator only works for forward pass.
  """
  assert not torch.is_grad_enabled()

  def wrapper(_func):

    @wraps(_func)
    def inner(*args, **kwargs):
      global _XLA_COMPUTATION_CACHE

      flat_inputs, input_tree_spec = tree_flatten((args, kwargs))
      tensor_inputs = pick_tensor_inputs(flat_inputs)
      computation_inputs = [None] * len(tensor_inputs)

      # TODO: Decide what to include in the cache key.
      if use_cache and _XLA_COMPUTATION_CACHE.get(_func.__name__,
                                                  None) is not None:
        fn_computation, output_tree_spec, parameter_ids = _XLA_COMPUTATION_CACHE[_func.__name__]
        for i in range(len(tensor_inputs)):
          computation_inputs[parameter_ids[i]] = tensor_inputs[i]
      else:
        flat_fake_inputs = [make_fake_inputs(a) for a in flat_inputs]
        tensor_inputs_function = make_function_tensor_inputs(
            _func, flat_fake_inputs, input_tree_spec)
        fake_tensor_inputs = pick_tensor_inputs(flat_fake_inputs)
        for i in range(len(fake_tensor_inputs)):
          assert fake_tensor_inputs[i].shape == tensor_inputs[i].shape

        fake_outputs = tensor_inputs_function(*fake_tensor_inputs)
        flat_fake_outputs, output_tree_spec = tree_flatten(fake_outputs)

        fn_ctx = torch_xla._XLAC.lowering.LoweringContext("FnComputation")
        fn_ctx.set_name_string("fn_ctx")
        fn_ctx.build(flat_fake_outputs)
        fn_hlo = fn_ctx.hlo()
        # print(f"fn_hlo: {fn_ctx.hlo_text()}")
        parameter_ids = []
        for t in fake_tensor_inputs:
          param_id = fn_ctx.tensor_parameter_id(t)
          if param_id != -1:
            parameter_ids.append(param_id)
        for i in range(len(fake_tensor_inputs)):
          computation_inputs[parameter_ids[i]] = tensor_inputs[i]

        fn_computation = xb.computation_from_module_proto(
            f"xla::xb_computation_{_func.__name__}", fn_hlo)
        if use_cache:
          _XLA_COMPUTATION_CACHE[_func.__name__] = (fn_computation,
                                                 output_tree_spec, parameter_ids)

      result = torch_xla._XLAC._xla_user_computation(
          f"xla::xb_computation_{_func.__name__}", computation_inputs, fn_computation)
      result_tree = tree_unflatten(result, output_tree_spec)
      return result_tree

    return inner

  if func is None:
    # Decorator was called with arguments, e.g., @assume_pure_torch(use_cache=True)
    # Return the actual decorator that will take the function
    return wrapper
  else:
    # Decorator was called without arguments, e.g., @assume_pure_torch
    # Call the wrapper directly with the function
    return wrapper(func)
