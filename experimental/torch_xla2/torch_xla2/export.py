# pylint: disable
"""Utilities for exporting a torch program to jax/stablehlo."""

import copy
from typing import Any, Dict, Tuple
import jax
import torch
from torch.fx import _pytree as fx_pytree
from torch_xla2 import ops_registry, tensor
from torch.utils import _pytree as pytree


class JaxProgram:
  def _wrap_inputs(self, xs, allow_torch_tensor=False):
    def convert(t):
      if isinstance(t, tensor.XLATensor2):
        return t
      if isinstance(t, torch.Tensor):
        if allow_torch_tensor:
          return tensor.move_to_device(t)
        else:
          raise ValueError("Regular torch.Tensor is not allowed.")
      if isinstance(t, jax.Array):
        return tensor.XLATensor2(t)
      return t

    return jax.tree_util.tree_map(convert, xs)

  def _unwrap_outputs(self, xs):
    def convert(t):
      if isinstance(t, tensor.XLATensor2):
        return t.jax()
      if isinstance(t, torch.Tensor):
        raise ValueError("Regular torch.Tensor is not allowed.")
      return t

    return jax.tree_util.tree_map(convert, xs)

  def __init__(
    self,
    exported_program,
    param_buffer_values,
    ordered_tensor_constants,
  ):
    self.param_buffer_values = self._wrap_inputs(
      param_buffer_values, allow_torch_tensor=True
    )
    self.ordered_tensor_constants = self._wrap_inputs(
      ordered_tensor_constants, allow_torch_tensor=True
    )
    self.exported_program = exported_program

  def __hash__(self):
    return hash(self.exported_program)

  @property
  def example_inputs(self):
    args, kwargs = self.exported_program.example_inputs
    args = pytree.tree_map(tensor.t2j, args)
    kwargs = pytree.tree_map(tensor.t2j, kwargs)
    return args, kwargs

  def flatten_inputs(self, args, kwargs):
    if args is None:
      args = tuple()
    if kwargs is None:
      kwargs = {}

    if (in_spec := self.exported_program.call_spec.in_spec) is not None:
      if (
        in_spec.type == tuple
        and len(in_spec.children_specs) == 2
        and in_spec.children_specs[0].type == tuple
        and in_spec.children_specs[1].type == dict
      ):
        # NOTE: this is the case where in_spec is for both args and kwargs
        return fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
      return fx_pytree.tree_flatten_spec(args, in_spec)
    return copy.deepcopy(args)

  def unflatten_outputs(self, res):
    return pytree.tree_unflatten(res, self.exported_program.call_spec.out_spec)

  def __call__(self, *args, **kwargs):
    inputs = self.flatten_inputs(args, kwargs)
    res = self.flatten_callable(*inputs)
    res = self.unflatten_outputs(res)

    return res

  @property
  def flatten_callable(self):
    def func(*inputs: jax.Array):
      nonlocal self
      inputs = self._wrap_inputs(inputs)
      num_mutations = len(
        self.exported_program.graph_signature.buffers_to_mutate
      )
      res = torch.fx.Interpreter(self.exported_program.graph_module).run(
        *self.param_buffer_values,
        *inputs,
        *self.ordered_tensor_constants,
        enable_io_processing=False,
      )
      res = res[num_mutations:]
      res = self._unwrap_outputs(res)
      return res

    return func

  def jit(self, *args, **kwargs):
    """Returns `jax.jit(self, *args, **kwargs)`."""
    return jax.jit(self, *args, **kwargs)

  def jit_lower(self, *args, **kwargs):
    """Returns `jax.jit(self, *args, **kwargs).lower(...)` with example_inputs used in export."""
    example_args, example_kwargs = self.example_inputs
    return self.jit(*args, **kwargs).lower(*example_args, **example_kwargs)


def exported_program_to_jax_program(ep):
  """exported_program_to_jax_program.

  Args:
    ep: torch.export.ExportedProgram

  Returns:
    JaxProgram

  """
  if torch.__version__ >= "2.2":
    ep = ep.run_decompositions()

  param_buffer_keys = ep.graph_signature.parameters + ep.graph_signature.buffers
  param_buffer_values = tuple(ep.state_dict[key] for key in param_buffer_keys)

  if hasattr(ep.graph_signature, "lifted_tensor_constants"):
    ordered_tensor_constants = tuple(
      ep.tensor_constants[name]
      for name in ep.graph_signature.lifted_tensor_constants
    )
  else:
    ordered_tensor_constants = tuple()

  return JaxProgram(ep, param_buffer_values, ordered_tensor_constants)


DEBUG = False


class JaxInterpreter(torch.fx.Interpreter):
  """Experimental."""

  def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:
    if not isinstance(
      target, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
    ):
      return super().call_function(target, args, kwargs)

    if DEBUG:
      print("Running ", target.name(), "--------")

    op = ops_registry.lowerings.lookup(target)
    if op is None:
      print(target.name(), target.tags)
      raise RuntimeError("No lowering found for", target.name())
    return op.func(*args, **kwargs)

  def run_node(self, n) -> Any:
    res = super().run_node(n)
    if DEBUG:
      if n.op == "call_function":
        if hasattr(res, "shape"):
          print("Meta:", n.meta.get("val").shape, "REAL: ", res.shape)
    return res


from torch._decomp import get_decompositions
import torch._refs

_extra_decomp = get_decompositions([torch.ops.aten.unfold])


def _extract_states_from_exported_program(exported_model):
  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = (
    exported_model.graph_signature.parameters
    + exported_model.graph_signature.buffers
  )
  state_dict = copy.copy(exported_model.state_dict)
  if (constants := getattr(exported_model, "constants", None)) is not None:
    state_dict.update(constants)
  param_buffer_values = list(state_dict[key] for key in param_and_buffer_keys)

  if hasattr(exported_model.graph_signature, "lifted_tensor_constants"):
    for name in exported_model.graph_signature.lifted_tensor_constants:
      param_buffer_values.append(exported_model.tensor_constants[name])

  return param_and_buffer_keys, param_buffer_values


def exported_program_to_jax(exported_program, export_raw: bool = False):
  """returns a pytree of jax arrays(state), and

  a callable(func) that is jax function.

  func(state, input) would be how you call it.
  """
  if torch.__version__ >= "2.2":
    # torch version 2.1 didn't expose this yet
    exported_program = exported_program.run_decompositions()
    exported_program = exported_program.run_decompositions(_extra_decomp)
  if DEBUG:
    print(exported_program.graph_module.code)

  names, states = _extract_states_from_exported_program(exported_program)

  def _extract_args(args, kwargs):
    flat_args, received_spec = pytree.tree_flatten((args, kwargs))  # type: ignore[possibly-undefined]
    return flat_args

  num_mutations = len(exported_program.graph_signature.buffers_to_mutate)

  def func(states, inputs):
    args = _extract_args(inputs, {})
    res = JaxInterpreter(exported_program.graph_module).run(
      *states,
      *args,
      enable_io_processing=False,
    )
    res = res[num_mutations:]
    return res

  if export_raw:
    return names, states, func

  states = pytree.tree_map_only(torch.Tensor, tensor.t2j, states)
  return states, func
