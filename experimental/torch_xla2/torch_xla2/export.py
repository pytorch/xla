# pylint: disable
"""Utilities for exporting a torch program to jax/stablehlo."""
import copy
from typing import Any, Dict, Tuple
import torch
from torch_xla2.ops import ops_registry
from torch_xla2 import tensor
from torch.utils import _pytree as pytree



DEBUG = False


class JaxInterpreter(torch.fx.Interpreter):
  """Experimental."""

  def __init__(self, graph_module):
    super().__init__(graph_module)
    import torch_xla2.ops.jaten
    import torch_xla2.ops.jtorch

  def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:
    if not isinstance(target,
                      (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
      return super().call_function(target, args, kwargs)

    if DEBUG:
      print('Running ', target.name(), '--------')

    op = ops_registry.all_aten_ops.get(target)
    if op is None:
      op = ops_registry.all_aten_ops.get(target.overloadpacket)
    if op is None:
      print(target.name(), target.tags)
      raise RuntimeError('No lowering found for', target.name())
    return op.func(*args, **kwargs)

  def run_node(self, n) -> Any:
    res = super().run_node(n)
    if DEBUG:
      if n.op == 'call_function':
        if hasattr(res, 'shape'):
          print('Meta:', n.meta.get('val').shape, 'REAL: ', res.shape)
    return res


from torch._decomp import get_decompositions
import torch._refs

_extra_decomp = get_decompositions([torch.ops.aten.unfold])


def _extract_states_from_exported_program(exported_model):
  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = exported_model.graph_signature.parameters + exported_model.graph_signature.buffers
  state_dict = copy.copy(exported_model.state_dict)
  if (constants := getattr(exported_model, 'constants', None)) is not None:
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
  if torch.__version__ >= '2.2':
    # torch version 2.1 didn't expose this yet
    exported_program = exported_program.run_decompositions()
    exported_program = exported_program.run_decompositions(_extra_decomp)
  if DEBUG:
    print(exported_program.graph_module.code)

  names, states = _extract_states_from_exported_program(exported_program)

  def _extract_args(args, kwargs):
    flat_args, received_spec = pytree.tree_flatten(
        (args, kwargs))  # type: ignore[possibly-undefined]
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
