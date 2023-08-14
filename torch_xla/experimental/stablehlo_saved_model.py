import copy
from dataclasses import dataclass
import enum
import itertools
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
from torch.utils._pytree import tree_map_only

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

  def __init__(self, bundle, default_method_name='forward'):
    self._bundle = bundle
    self._name_to_stablehlo = {
        meta.name: (meta, stablehlo)
        for meta, stablehlo in bundle.stablehlo_funcs
    }
    self._default_method = default_method_name

  def evaluate(self, method_name, args):
    from tensorflow.compiler.tf2xla.python import xla as tfxla
    meta, stablehlo = self._name_to_stablehlo[method_name]
    call_args = []
    for loc in meta.input_locations:
      if loc.type_ == VariableType.PARAMETER:
        call_args.append(self._bundle.state_dict[loc.name])
      elif loc.type_ == VariableType.CONSTANT:
        call_args.append(self._bundle.additional_constants[loc.position])
      else:
        call_args.append(args[loc.position])
    output_sig = meta.output_signature[0]
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=[output_sig.dtype],  # dtype information
        Sout=[output_sig.shape],  # Shape information
        function_list=[],
        platforms=('CPU',),
        module=stablehlo,
    )

  def __call__(self, *args):
    return self.evaluate(self._default_method, args)


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

  device = xm.xla_device()
  args = tuple(
      tree_map_only(torch.Tensor, lambda x: x.to(device=device), sample_inputs))
  orig_state_dict = copy.copy(model.state_dict())
  model = model.to(device)
  bundle = _callable_to_stablehlo_bundle(model, args, model.state_dict())
  bundle.state_dict = orig_state_dict
  stablehlo_model = SHLOModel(bundle)
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
  stablehlo_version: str
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


def _callable_to_stablehlo_bundle(func, input_args, state_dict):
  xm.mark_step()
  xm.wait_device_ops()
  metrics.clear_counters()
  device = xm.xla_device()
  res = func(*input_args)

  # If there are any fallback ops, this means that in torch/XLA side,
  # not all ops are lowerable to HLO.
  if fallback_ops := dynamo_bridge.get_fallback_ops():
    message = "This model contains ops not capturable by Pytorch/XLA: {}".format(
        "\n".join(fallback_ops))
    raise RuntimeError(message)

  if isinstance(res, torch.Tensor):
    res = (res,)

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(res)

  tensor_id_to_state_name = {
      torch_xla._XLAC._xla_get_tensor_id(value): name
      for name, value in state_dict.items()
      if isinstance(value, torch.Tensor)
  }

  stablehlo_content = xm.get_stablehlo_bytecode(res)

  pos_to_orig_pos = {}
  pos_to_param = {}
  input_locations = []
  input_signatures = []
  additional_constants = []
  input_ids = {
      torch_xla._XLAC._xla_get_tensor_id(tensor): pos
      for pos, tensor in enumerate(input_args)
      if isinstance(tensor, torch.Tensor)
  }
  for hlo_input_pos, (tensor_id, tensor_value) in enumerate(
      zip(graph_input_tensor_ids, graph_input_xla_values)):
    if tensor_id in input_ids:  # this is input
      location = InputLocation.input_arg(position=input_ids[tensor_id])
    elif tensor_id in tensor_id_to_state_name:
      location = InputLocation.parameter(
          name=tensor_id_to_state_name[tensor_id])
    else:  # additional constants that WE created
      location = InputLocation.constant(position=len(additional_constants))
      additional_constants.append(tensor_value.detach().cpu().numpy())
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

  torch_xla._XLAC._clear_pending_irs(str(device))

  meta = StableHLOFunctionMeta(
      name='forward',
      # TODO(qihqi) populate version from runtime
      stablehlo_version="0.0.0",
      input_signature=input_signatures,
      output_signature=output_signature,
      input_locations=input_locations,
  )

  return StableHLOModelBundle(
      stablehlo_funcs=[(meta, stablehlo_content)],
      state_dict={},
      additional_constants=additional_constants,
  )


class XLAExportInterpreter(torch.fx.Interpreter):

  def __init__(self, module, device):
    self._device = device
    super().__init__(module)

  def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:
    # NOTE(qihqi): We need to do this because there are some operators
    # that creates new tensor. And those operators would create it on CPU.
    # this bit of code basically move it to XLA device, otherwise we would
    # get an error saying we cannot do math between a XLA tensor and a CPU
    # tensor.
    new_kwargs = dict(kwargs)
    if 'device' in kwargs:
      new_kwargs['device'] = self._device
    return super().call_function(target, args, new_kwargs)


def _exported_program_to_stablehlo_bundle(exported_model,
                                          args) -> StableHLOModelBundle:
  device = xm.xla_device()

  if exported_model.call_spec.in_spec is not None:
    args = fx_pytree.tree_flatten_spec(args, exported_model.call_spec.in_spec)
  else:
    args = copy.deepcopy(args)

  args = tree_map_only(torch.Tensor, lambda x: x.to(device=device), args)

  # NOTE call convention: (parameters, buffers, user_inputs)
  param_and_buffer_keys = exported_model.graph_signature.parameters + exported_model.graph_signature.buffers
  state = tree_map_only(torch.Tensor, lambda x: x.to(device=device),
                        exported_model.state_dict)
  param_buffer_values = (state[key] for key in param_and_buffer_keys)

  num_mutations = len(exported_model.graph_signature.buffers_to_mutate)

  def forward(*args):
    with torch.no_grad():
      res = XLAExportInterpreter(exported_model.graph_module, device).run(
          *param_buffer_values, *args, enable_io_processing=False)
      return res[num_mutations:]

  bundle = _callable_to_stablehlo_bundle(forward, args, state)
  bundle.state_dict = tree_map_only(torch.Tensor,
                                    lambda x: x.detach().cpu().numpy(),
                                    exported_model.state_dict)
  return bundle


class StableHLOExportOptions:
  pass


def _save_program_bundle(bundle: StableHLOModelBundle,
                         stablehlo_dir: os.PathLike) -> None:

  data_dir = os.path.join(stablehlo_dir, 'data')
  os.makedirs(data_dir, exist_ok=True)
  for key, val in bundle.state_dict.items():
    with open(os.path.join(stablehlo_dir, 'data', key), 'wb') as f:
      np.save(f, val)

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
      np.save(f, constant)


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
