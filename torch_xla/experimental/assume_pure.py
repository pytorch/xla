from copy import copy
from functools import wraps
from typing import Dict

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


def make_fake_inputs(input):
  """Creates a fake input for the given input torch tensor. If the input
  is not a tensor, it returns the input as is.
  """
  if isinstance(input, torch.Tensor):
    t = xb.create_placeholder_tensor(input.shape, input.dtype)
    return t.requires_grad_(input.requires_grad)
  return input


def prepare_computation_inputs(fn_ctx, flat_fake_inputs, flat_inputs):
  """Prepares the computation inputs for the XLA computation.

  fn_ctx contains the mapping fake tensors in flat_fake_inputs to the input 
  parameter id in xla computation. We use this mapping to pick actual inputs
  from flat_inputs to create the computation inputs.
  
  Args:
  fn_ctx: The lowering context for the function.
  flat_fake_inputs: The flattened fake inputs for the function.
  flat_inputs: The flattened actual inputs for the function.
  Returns:
  computation_inputs: The computation inputs for the XLA computation.
  hoisted_vars_map: The hoisted variables map for the XLA computation.
  hlo_input_id_to_input_index_map: The mapping from HLO input IDs to input
    indices for flat_inputs.
  """
  all_hlo_input_vars_map: Dict[
      int, torch.Tensor] = fn_ctx.device_parameter_id_tensor_mapping()
  hlo_input_id_to_input_index_map: Dict[int, int] = {}
  computation_inputs = [None] * len(all_hlo_input_vars_map)
  for i, t in enumerate(flat_fake_inputs):
    if isinstance(t, torch.Tensor):
      param_id = fn_ctx.tensor_parameter_id(t)
      if param_id != -1:
        computation_inputs[param_id] = flat_inputs[i]
        hlo_input_id_to_input_index_map[param_id] = i
        del all_hlo_input_vars_map[param_id]

  # The remaining variables in all_input_vars_map are the hoisted variables
  # that are not present in flat_inputs.
  hoisted_vars_map = all_hlo_input_vars_map
  for i, var in hoisted_vars_map.items():
    computation_inputs[i] = var

  return computation_inputs, hoisted_vars_map, hlo_input_id_to_input_index_map


def assume_pure_torch(func, use_cache=False):
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

  @wraps(func)
  def inner(*args, **kwargs):
    global _XLA_COMPUTATION_CACHE

    flat_inputs, input_tree_spec = tree_flatten((args, kwargs))
    computation_inputs = None

    # TODO: Decide what to include in the cache key.
    if use_cache and _XLA_COMPUTATION_CACHE.get(func.__name__,
                                                None) is not None:
      fn_computation, output_tree_spec, hlo_input_id_to_input_index_map, hoisted_vars = _XLA_COMPUTATION_CACHE[
          func.__name__]
      computation_inputs_size = len(hoisted_vars) + len(
          hlo_input_id_to_input_index_map)
      computation_inputs = [None] * (computation_inputs_size)
      for hlo_id, input_index in hlo_input_id_to_input_index_map.items():
        computation_inputs[hlo_id] = flat_inputs[input_index]
      for i, var in hoisted_vars.items():
        computation_inputs[i] = var
    else:
      flat_fake_inputs = [make_fake_inputs(a) for a in flat_inputs]
      fake_args, fake_kwargs = tree_unflatten(flat_fake_inputs, input_tree_spec)
      fake_outputs = func(*fake_args, **fake_kwargs)
      flat_fake_outputs, output_tree_spec = tree_flatten(fake_outputs)

      fn_ctx = torch_xla._XLAC.lowering.LoweringContext("FnComputation")
      fn_ctx.set_name_string("fn_ctx")
      fn_ctx.build(flat_fake_outputs)
      fn_hlo = fn_ctx.hlo()
      fn_computation = xb.computation_from_module_proto(
          f"xla::xb_computation_{func.__name__}", fn_hlo)

      computation_inputs, hoisted_vars, hlo_input_id_to_input_index_map = prepare_computation_inputs(
          fn_ctx, flat_fake_inputs, flat_inputs)

      if use_cache:
        _XLA_COMPUTATION_CACHE[func.__name__] = (
            fn_computation, output_tree_spec, hlo_input_id_to_input_index_map,
            hoisted_vars)

    result = torch_xla._XLAC._xla_user_computation(
        f"xla::xb_computation_{func.__name__}", computation_inputs,
        fn_computation)
    result_tree = tree_unflatten(result, output_tree_spec)
    return result_tree

  return inner
