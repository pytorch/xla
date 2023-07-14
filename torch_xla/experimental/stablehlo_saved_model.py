import copy
from dataclasses import dataclass
import enum
import json
import shutil
import os
import re
from typing import List, Tuple, Optional, Mapping, Any, Dict
import dataclasses

import numpy as np
import torch
from torch import nn
from torch.fx import _pytree as fx_pytree
import torch_xla
from torch_xla.core import xla_model as xm
from torch_xla.core import dynamo_bridge
from torch_xla.debug import metrics
import torchvision
import torch._dynamo as torchdynamo

from typing import Tuple, Type, Callable

import sys


def _get_numpy_dtype(dtype):
  return {
      torch.double: np.double,
      torch.float32: np.float32,
      torch.half: np.half,
      torch.long: np.int64,
      torch.int32: np.int32,
      torch.int16: np.int16,
      torch.bool: np.bool_,
  }.get(dtype)


class SHLOModel:

  def __init__(self, model_bytecode, pos_to_orig_pos, pos_to_param,
               output_shape, output_dtype):
    self.model_bytecode = model_bytecode
    self.pos_to_orig_pos = pos_to_orig_pos
    self.pos_to_param = pos_to_param
    self._total_number = len(pos_to_orig_pos) + len(pos_to_param)

    self.tout = (_get_numpy_dtype(output_dtype),)
    self.sout = (tuple(output_shape),)

  def __call__(self, *args):
    # TODO(qihqi): remove this
    from tensorflow.compiler.tf2xla.python import xla as tfxla
    call_args = []
    for i in range(self._total_number):
      if i in self.pos_to_orig_pos:
        call_args.append(args[self.pos_to_orig_pos[i]])
      else:
        call_args.append(self.pos_to_param[i])

    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=self.tout,  # dtype information
        Sout=self.sout,  # Shape information
        function_list=[],
        platforms=('CPU',),
        module=self.model_bytecode,
    )[0]


def export_torch_model(model: torch.nn.Module,
                       sample_inputs: Tuple[torch.Tensor]):
  """Convert model into a callable backed by StableHLO.

  Args:
      model: torch.nn.Module - a pytorch model
      sample_inputs: Tuple[torch.Tensor] - The input to this model
      output_shape: Tuple[int] - Shape of the output
      output_dtype: dtype - numpy dtype for the output

  This function will return a callable backed by StableHLO such that,

  model(*sample_inputs) ==  export_torch_model(model, sample_inputs)(*sample_inputs)
  (up to numerics)

  In other words, returned callable have the same calling convention of the input model, and
  on the sample input, or inputs sufficiently similar* to sample input,
  it is will to return same result as the original model.

  * sufficiently similar input because this function will use tracing to extract the model operations
    so it might specialize on the shapes of the sample input.

  For now, model has to only take Tensors as input and has to return 1 tensor as output.

  """

  # Materialize the computation before this call to make sure that StableHLO
  # captured below only reflect the model passed in
  xm.mark_step()
  metrics.clear_counters()
  model = copy.deepcopy(model)
  sample_inputs = copy.deepcopy(sample_inputs)

  device = xm.xla_device()
  model = model.to(device=device)
  sample_inputs_lazy = tuple(
      x.to(device=device) if isinstance(x, torch.Tensor) else x
      for x in sample_inputs)
  input_ids = {
      torch_xla._XLAC._xla_get_tensor_id(tensor): i
      for i, tensor in enumerate(sample_inputs_lazy)
      if isinstance(tensor, torch.Tensor)
  }
  output = model(*sample_inputs_lazy)
  if fallback_ops := dynamo_bridge.get_fallback_ops():
    message = "This model contains ops not capturable by Pytorch/XLA: {}".format(
        "\n".join(fallback_ops))
    raise RuntimeError(message)
  stablehlo_bytecode = xm.get_stablehlo_bytecode([output])

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node([output])

  pos_to_orig_pos = {}
  pos_to_param = {}
  for hlo_input_pos, tensor_id in enumerate(graph_input_tensor_ids):
    if tensor_id in input_ids:  # this is input
      pos_to_orig_pos[hlo_input_pos] = input_ids[tensor_id]
    else:
      pos_to_param[hlo_input_pos] = graph_input_xla_values[
          hlo_input_pos].detach().cpu().numpy()

  stablehlo_model = SHLOModel(stablehlo_bytecode, pos_to_orig_pos, pos_to_param,
                              output.shape, output.dtype)
  # Remove all of the pending IR from all live tensors. The assumptions are
  # 1. With the  `xm.mark_step` in the beginning of this call, every XLATensor
  # should be materialized
  # 2. All of the pending IRs are result of the one inference
  # should be removed to avoid extra computation executed and in place updates op
  # mistakenlly update the input tensors.
  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))
  return stablehlo_model


class VariableType(enum.Enum):
  INPUT_ARG = 'input_arg'
  PARAMETER = 'parameter'
  CONSTANT = 'constant'


@dataclass
class VariableSignature:  # either argument or parameters
  shape: List[int]
  dtype: str


@dataclass
class InputLocation:
  type_: VariableType
  position: int = -1
  name: str = ''

  @classmethod
  def parameter(cls, name: str):
    return cls(type_=VariableType.PARAMETER, name=name)

  @classmethod
  def input_arg(cls, position: int):
    return cls(type_=VariableType.INPUT_ARG, position=position)

  @classmethod
  def constant(cls, position):
    return cls(type_=VariableType.CONSTANT, position=position)


@dataclass
class StableHLOFunctionMeta:
  # name of the callable.
  name: str
  # version.
  stablehlo_version: int
  # order is the order accepted by the stablehlo
  input_signature: List[VariableSignature]
  output_signature: List[VariableSignature]
  # An input to the underlying stable callable can come from
  # the arguments the user supplied, OR a parameter, OR a constant
  input_locations: List[InputLocation]


class StableHLOJSONSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj)
    if isinstance(obj, VariableType):
      return obj.value
    return super().default(obj)


def stablehlo_obj_hook(dct):
  targets = [
      StableHLOFunctionMeta,
      VariableSignature,
      InputLocation,
      VariableSignature,
  ]

  def _match_field(clazz):
    # A dataclass become a dict on serialization then,
    # it the dict must have values for all the fields in that dataclass.
    return set(f.name for f in dataclasses.fields(clazz)) == dct.keys()

  def _try_convert_as_enum(v):
    try:
      return VariableType(v)
    except:
      return v

  for clazz in targets:
    if _match_field(clazz):
      new_dict = {k: _try_convert_as_enum(v) for k, v in dct.items()}
      return clazz(**new_dict)


@dataclass
class StableHLOModelBundle:
  # original state dict; but torch.Tensor's converted to np.array
  state_dict: Dict[str, Any]
  # Additional constants that we decide to hardcode.
  additional_constants: List[np.ndarray]
  # can support the case of multiple callable of the same model.
  stablehlo_funcs: List[Tuple[StableHLOFunctionMeta, bytes]]


@dataclass
class StableHLOExportOptions:
  pass


def _exported_program_to_stablehlo_bundle(exported_model, args):
  xm.mark_step()
  metrics.clear_counters()
  device = xm.xla_device()

  if exported_model.call_spec.in_spec is not None:
    args = fx_pytree.tree_flatten_spec(args, exported_model.call_spec.in_spec)
  else:
    args = copy.deepcopy(args)

  args = [
      x.to(device=device) if isinstance(x, torch.Tensor) else x for x in args
  ]

  input_ids = {
      torch_xla._XLAC._xla_get_tensor_id(tensor): i
      for i, tensor in enumerate(args)
      if isinstance(tensor, torch.Tensor)
  }
  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = exported_model.graph_signature.parameters + exported_model.graph_signature.buffers
  state = exported_model.state_dict
  param_buffer_values = tuple(state[key].to(
      device=device) if isinstance(state[key], torch.Tensor) else state[key]
                              for key in param_and_buffer_keys)

  with torch.no_grad():
    res = torch.fx.Interpreter(exported_model.graph_module).run(
        *param_buffer_values, *args, enable_io_processing=False)

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(res)

  tensor_id_to_state_name = {
      torch_xla._XLAC._xla_get_tensor_id(value): name
      for name, value in zip(param_and_buffer_keys, param_buffer_values)
      if isinstance(value, torch.Tensor)
  }
  stablehlo_content = xm.get_stablehlo_bytecode(res)

  pos_to_orig_pos = {}
  pos_to_param = {}
  input_locations = []
  input_signatures = []
  additional_constants = []
  for hlo_input_pos, (tensor_id, tensor_value) in enumerate(
      zip(graph_input_tensor_ids, graph_input_xla_values)):
    if tensor_id in input_ids:  # this is input
      location = InputLocation.input_arg(position=input_ids[tensor_id])
    elif tensor_id in tensor_id_to_state_name:
      location = InputLocation.parameter(
          name=tensor_id_to_state_name[tensor_id])
    else:  # additional constants that WE created
      location = InputLocation.constant(position=len(additional_constants))
      additional_constants.append(tensor_value)
    input_locations.append(location)
    input_signatures.append(
        VariableSignature(
            shape=list(tensor_value.shape),
            dtype=str(tensor_value.dtype).replace('torch.', '')))

  output_signature = [
      VariableSignature(
          shape=list(tensor.shape),
          dtype=str(tensor_value.dtype).replace('torch.', '')) for tensor in res
  ]

  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))

  meta = StableHLOFunctionMeta(
      name='forward',
      stablehlo_version=5,
      input_signature=input_signatures,
      output_signature=output_signature,
      input_locations=input_locations,
  )

  return StableHLOModelBundle(
      stablehlo_funcs=[(meta, stablehlo_content)],
      state_dict=exported_model.state_dict,
      additional_constants=additional_constants,
  )


class StableHLOExportOptions:
  pass


def _save_program_bundle(bundle: StableHLOModelBundle,
                         stablehlo_dir: os.PathLike) -> None:

  data_dir = os.path.join(stablehlo_dir, 'data')
  os.makedirs(data_dir, exist_ok=True)
  for key, val in bundle.state_dict.items():
    with open(os.path.join(stablehlo_dir, 'data', key), 'wb') as f:
      np.save(f, val.cpu().detach().numpy())

  # save metadata and stablehlo bytecode
  func_dir = os.path.join(stablehlo_dir, 'functions')
  os.makedirs(func_dir, exist_ok=True)
  for meta, bytecode in bundle.stablehlo_funcs:
    with open(os.path.join(func_dir, meta.name + '.meta'), 'w') as f:
      json.dump(meta, f, cls=StableHLOJSONSerializer)
    with open(os.path.join(func_dir, meta.name + '.mlir'), 'wb') as f:
      f.write(bytecode)

  const_dir = os.path.join(stablehlo_dir, 'constants')
  os.makedirs(const_dir, exist_ok=True)
  for i, constant in enumerate(bundle.additional_constants):
    with open(os.path.join(const_dir, str(i)), 'wb') as f:
      np.save(f, constant.cpu().detach().numpy())


def _iter_dir(path: os.PathLike):
  for name in os.listdir(path):
    with open(os.path.join(path, name), 'rb') as f:
      yield name, f


def _load_program_bundle(stablehlo_dir: os.PathLike) -> StableHLOModelBundle:
  state_dict = {}
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'data')):
    state_dict[name] = np.load(f, allow_pickle=True)

  constants = []
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'constants')):
    # name of constants are ints
    constants.append((int(name), np.load(f, allow_pickle=True)))
  constants = [v for k, v in sorted(constants)]

  metas = []
  name_to_bytecode = {}
  stablehlo_funcs = []
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'functions')):
    if name.endswith('.meta'):
      metas.append(json.load(f, object_hook=stablehlo_obj_hook))
    else:
      name_to_bytecode[os.path.splitext(name)[0]] = f.read()

  for meta in metas:
    stablehlo_funcs.append((meta, name_to_bytecode[meta.name]))

  return StableHLOModelBundle(
      stablehlo_funcs=stablehlo_funcs,
      additional_constants=constants,
      state_dict=state_dict)


def save_as_stablehlo(exported_model: 'ExportedProgram',
                      args: Tuple[Any],
                      stablehlo_dir: os.PathLike,
                      options: Optional[StableHLOExportOptions] = None):

  bundle = _exported_program_to_stablehlo_bundle(exported_model, args)
  _save_program_bundle(bundle, stablehlo_dir)
