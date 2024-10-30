import copy
import dataclasses
import enum
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import torch
from torch.fx import subgraph_rewriter
import torch_xla
import torch_xla.experimental.quantized
from torch._decomp import get_decompositions, core_aten_decompositions
from torch._export.serde.serialize import GraphModuleSerializer
from torch.fx import _pytree as fx_pytree
from torch.utils import _pytree as pytree
from torch_xla._dynamo import dynamo_bridge
from torch_xla.core import xla_model as xm
from torch_xla.debug import metrics
from torch_xla.experimental.stablehlo_custom_call import (
    extract_custom_call_outputs_shape_dtype, stablehlo_custom_call)
from torch_xla.experimental.unbounded_dynamism_export import (
    exported_program_has_symbolic_input_shape,
    process_exported_program_with_symbolic_input)


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


def _extract_call_parameters(args, meta, bundle):
  call_args = []
  if meta.input_pytree_spec is not None:
    args, _ = pytree.tree_flatten(args)
  for loc in meta.input_locations:
    if loc.type_ == VariableType.PARAMETER:
      call_args.append(bundle.state_dict[loc.name])
    elif loc.type_ == VariableType.CONSTANT:
      call_args.append(bundle.additional_constants[loc.position])
    else:
      call_args.append(args[loc.position])
  return call_args


@dataclass
class StableHLOExportOptions:
  include_human_readable_text: bool = True
  override_tracing_arguments: Optional[Tuple[Any]] = None
  override_tracing_kwargs: Optional[Mapping[str, Any]] = None
  save_weights: bool = True
  # Inline all constants in StableHLO if True. Otherwise only
  # special constants (0 and 1) are inlined.
  inline_all_constant: bool = True

  # Whether to export the weights
  export_weights: bool = True
  # Ops that will be mapped to stablehlo.custom_call in the
  # exported StableHLO graph.
  custom_ops_allowed_in_graph: Set[str] = field(default_factory=set)
  # Export node metadata to NamedLoc in StableHLO.
  export_node_metadata: bool = False


class StableHLOGraphModule:

  def __init__(self, bundle):
    self._bundle = bundle
    self._name_to_stablehlo = {
        func.meta.name: func for func in bundle.stablehlo_funcs
    }
    self._default_method = bundle.stablehlo_funcs[0].meta.name

  def evaluate(self, method_name, args):
    func = self._name_to_stablehlo[method_name]
    call_args = _extract_call_parameters(args, func.meta, self._bundle)
    call_args = [
        torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        for x in call_args
    ]
    res = torch_xla._XLAC._run_stablehlo(func.bytecode, call_args)
    if func.meta.output_pytree_spec is not None:
      out_spec = pytree.treespec_loads(func.meta.output_pytree_spec)
      res = pytree.tree_unflatten(res, out_spec)
    return res

  def get_stablehlo_bytecode(self, method_name=None):
    if method_name is None:
      method_name = self._default_method
    return self._name_to_stablehlo[method_name].bytecode

  def get_stablehlo_text(self, method_name=None):
    if method_name is None:
      method_name = self._default_method
    return self._name_to_stablehlo[method_name].text

  def save(self,
           directory_path,
           options: Optional[StableHLOExportOptions] = None):
    _save_program_bundle(self._bundle, directory_path, options)

  @classmethod
  def load(cls, directory_path):
    bundle = _load_program_bundle(directory_path)
    return cls(bundle)

  def __call__(self, *args):
    return self.evaluate(self._default_method, args)


class VariableType(enum.Enum):
  INPUT_ARG = 'input_arg'
  PARAMETER = 'parameter'
  CONSTANT = 'constant'


@dataclass
class VariableSignature:  # either argument or parameters
  shape: List[int]
  dtype: str
  dynamic_dims: List[int] = dataclasses.field(default_factory=list)


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

  unused_inputs: List[Tuple[InputLocation, VariableSignature]]

  # input_pytree_spec
  input_pytree_spec: Optional[str] = None
  output_pytree_spec: Optional[str] = None


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
      StableHLOFunc,
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
class StableHLOFunc:
  meta: StableHLOFunctionMeta
  bytecode: bytes
  text: Optional[str]


@dataclass
class StableHLOModelBundle:
  # original state dict; but torch.Tensor's converted to np.array
  state_dict: Dict[str, Any]
  # Additional constants that we decide to hardcode.
  additional_constants: List[np.ndarray]
  # can support the case of multiple callable of the same model.
  stablehlo_funcs: List[StableHLOFunc]


class XLAExportInterpreter(torch.fx.Interpreter):

  def __init__(self, module, device, custom_ops_allowed_in_graph,
               gm_serializer):
    self._device = device
    super().__init__(module)
    self.tensor_id_to_dynamic_dims = {}
    self.custom_ops_allowed_in_graph = custom_ops_allowed_in_graph
    self.gm_serializer = gm_serializer

  def _mark_dynamic(self, tensor, dynamic_dims):
    tid = torch_xla._XLAC._xla_get_tensor_id(tensor)
    self.tensor_id_to_dynamic_dims[tid] = dynamic_dims
    for i in dynamic_dims:
      torch_xla._XLAC._xla_mark_dynamic(tensor, i)

  def call_function(self, target, args: Tuple, kwargs: Dict) -> Any:
    # NOTE(qihqi): We need to do this because there are some operators
    # that creates new tensor. And those operators would create it on CPU.
    # this bit of code basically move it to XLA device, otherwise we would
    # get an error saying we cannot do math between a XLA tensor and a CPU
    # tensor.
    new_kwargs = dict(kwargs)
    if 'device' in kwargs:
      new_kwargs['device'] = self._device
    # If the op spec expects a `Tensor` input, we wrap the python float/int
    # to a torch.tensor. The dtype for float respects
    # `torch.get_default_dtype`. Without this wrapping, the python float
    # will be wrapped before it enters dispatcher, and it doesn't respect
    # the global default dtype.
    if hasattr(target, '_schema'):
      # Note: Use `_disable_current_modes` to alwasys create constant tensor.
      # Under `fake_tensor_mode` a fake tensor will be created. This is not a
      # use case for XLAExportInterpreter right now, adding to be future-proof.
      with torch.utils._python_dispatch._disable_current_modes():
        args_and_specs = tuple(zip(args, target._schema.arguments))
        args = tuple(
            map(
                lambda arg_spec: torch.tensor(arg_spec[0])
                if isinstance(arg_spec[0], float) and type(arg_spec[1].type) ==
                torch.TensorType else arg_spec[0], args_and_specs))
    return super().call_function(target, args, new_kwargs)

  def run_node(self, n) -> Any:
    if n.op == 'placeholder':
      fake_t = n.meta['val']
      res = super().run_node(n)
      if hasattr(fake_t, 'shape'):
        dynamic_dims = [
            i for i, x in enumerate(fake_t.shape) if not isinstance(x, int)
        ]
        self._mark_dynamic(res, dynamic_dims)
    elif n.op == 'call_function':
      if hasattr(n.target, 'namespace'
                ) and n.target.namespace in self.custom_ops_allowed_in_graph:
        output_shapes, output_dtypes = extract_custom_call_outputs_shape_dtype(
            n)
        call_name = str(n.target)
        n.target = stablehlo_custom_call
        n.args = (n.args, call_name, output_shapes, output_dtypes)
      res = super().run_node(n)
    else:
      res = super().run_node(n)
    if self.gm_serializer is not None:
      from torch_xla.experimental.xla_mlir_debuginfo import write_mlir_debuginfo
      node_metadata = json.dumps(self.gm_serializer.serialize_metadata(n))
      pytree.tree_map_only(torch.Tensor,
                           lambda x: write_mlir_debuginfo(x, node_metadata),
                           res)
    return res


def _extract_input_args(exported_model, options):
  if options.override_tracing_arguments is not None:
    args = options.override_tracing_arguments
    kwargs = options.override_tracing_kwargs
  elif hasattr(exported_model, 'example_inputs'):
    args, kwargs = exported_model.example_inputs
  else:
    raise ValueError(
        'No argument is provided, please make sure that ExportedProgram.example_inputs() has content'
    )

  if (in_spec := exported_model.call_spec.in_spec) is not None:
    if (in_spec.type == tuple and len(in_spec.children_specs) == 2 and
        in_spec.children_specs[0].type == tuple and
        in_spec.children_specs[1].type == dict):
      # NOTE: this is the case where in_spec is for both args and kwargs
      return fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
    else:
      return fx_pytree.tree_flatten_spec(args, in_spec)
  else:
    return copy.deepcopy(args)


_extra_decompositions = get_decompositions([torch.ops.aten.grid_sampler_2d])


def _run_decompositions(exported_model):
  if hasattr(torch.export, 'default_decompositions'):
    decomp_table = torch.export.default_decompositions()
  else:
    decomp_table = core_aten_decompositions()
  decomp_table.update(_extra_decompositions)
  decomp_table[torch.ops.aten._safe_softmax.default] = torch.softmax
  exported_model = exported_model.run_decompositions(decomp_table)
  return exported_model


def _exported_program_to_stablehlo_bundle(exported_model,
                                          options) -> StableHLOModelBundle:
  exported_model = _run_decompositions(exported_model)
  if options is None:
    options = StableHLOExportOptions()
  if exported_program_has_symbolic_input_shape(exported_model):
    process_exported_program_with_symbolic_input(exported_model)
  args, kwargs = exported_model.example_inputs

  assert len(kwargs) == 0, "Export to stablehlo doesnt support kwargs yet."

  device = xm.xla_device()

  _flat_input_args = exported_model._graph_module_flat_inputs(args, {})
  _flat_input_args = pytree.tree_map_only(torch.Tensor,
                                          lambda x: x.to(device=device),
                                          _flat_input_args)

  num_mutations = len(exported_model.graph_signature.buffers_to_mutate)
  xm.mark_step()
  xm.wait_device_ops()
  metrics.clear_counters()
  device = xm.xla_device()

  # Run the fx graph tracing using lazy tensor
  if options.inline_all_constant:
    # Inline all constants.
    torch_xla._XLAC._set_xla_all_numbers_special_scalars(True)
  xla_hlo_debug_env = os.environ.get("XLA_HLO_DEBUG", "0")
  if options.export_node_metadata:
    gm_serializer = GraphModuleSerializer(exported_model.graph_signature,
                                          exported_model.module_call_graph)
    os.environ["XLA_HLO_DEBUG"] = "1"
  else:
    gm_serializer = None

  xla_interpreter = XLAExportInterpreter(exported_model.graph_module, device,
                                         options.custom_ops_allowed_in_graph,
                                         gm_serializer)
  with torch.no_grad():
    res = xla_interpreter.run(*_flat_input_args, enable_io_processing=False)
    res = res[num_mutations:]

  # If there are any fallback ops, this means that in torch/XLA side,
  # not all ops are lowerable to HLO.
  if fallback_ops := dynamo_bridge.get_fallback_ops():
    message = "This model contains ops not capturable by Pytorch/XLA: {}".format(
        "\n".join(fallback_ops))
    raise RuntimeError(message)

  InputKind = torch.export.graph_signature.InputKind
  tensor_id_to_state_name = {}
  state_dict = {}
  input_ids = {}
  for i, (tensor, input_spec) in enumerate(
      zip(_flat_input_args, exported_model.graph_signature.input_specs)):
    # Assumption:
    # All states comes first in the list of args, and user provided inputs comes later.
    # Also there is no kwargs.
    if not isinstance(tensor, torch.Tensor):
      continue

    tensor_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    if input_spec.kind == InputKind.USER_INPUT:
      input_position = i - len(state_dict)
      input_ids[tensor_id] = input_position
    else:
      state_dict[input_spec.target] = tensor
      tensor_id_to_state_name[tensor_id] = input_spec.target

  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(res)

  stablehlo_content = xm.get_stablehlo_bytecode(res)
  if options.include_human_readable_text:
    stablehlo_text = xm.get_stablehlo(res)
  else:
    stablehlo_text = None

  pos_to_orig_pos = {}
  pos_to_param = {}
  input_locations = []
  input_signatures = []
  additional_constants = []

  # there might be inputs that is part of input but not consumed by HLO graph
  unused_input_positions = set(range(len(args)))

  for hlo_input_pos, (tensor_id, tensor_value) in enumerate(
      zip(graph_input_tensor_ids, graph_input_xla_values)):
    if tensor_id in input_ids:  # this is input
      pos_id = input_ids[tensor_id]
      location = InputLocation.input_arg(position=pos_id)
      if pos_id in unused_input_positions:
        unused_input_positions.remove(pos_id)
    elif tensor_id in tensor_id_to_state_name:
      location = InputLocation.parameter(
          name=tensor_id_to_state_name[tensor_id])
    else:  # additional constants that WE created
      location = InputLocation.constant(position=len(additional_constants))
      additional_constants.append(tensor_value.detach().cpu().numpy())
    input_locations.append(location)
    dynamic_dims = xla_interpreter.tensor_id_to_dynamic_dims.get(tensor_id, [])
    input_signatures.append(
        VariableSignature(
            shape=list(tensor_value.shape),
            dtype=str(tensor_value.dtype).replace('torch.', ''),
            dynamic_dims=dynamic_dims))

  unused_inputs = []
  for i in unused_input_positions:
    pos = InputLocation.input_arg(position=i)
    arg = args[i]
    if isinstance(arg, torch.Tensor):
      signature = VariableSignature(
          shape=list(arg.shape), dtype=str(arg.dtype).replace('torch.', ''))
    else:
      signature = VariableSignature(
          shape=[],
          dtype=type(arg).__name__,
      )

    unused_inputs.append((pos, signature))

  output_signature = [
      VariableSignature(
          shape=list(tensor.shape),
          dtype=str(tensor.dtype).replace('torch.', '')) for tensor in res
  ]

  torch_xla._XLAC._clear_pending_irs(str(device))

  meta = StableHLOFunctionMeta(
      name='forward',
      # TODO(qihqi) populate version from runtime
      stablehlo_version="0.0.0",
      input_signature=input_signatures,
      output_signature=output_signature,
      input_locations=input_locations,
      unused_inputs=unused_inputs,
      input_pytree_spec=pytree.treespec_dumps(exported_model.call_spec.in_spec),
      output_pytree_spec=pytree.treespec_dumps(
          exported_model.call_spec.out_spec),
  )

  exported_state_dict = {}
  if options.export_weights:
    exported_state_dict = pytree.tree_map_only(
        torch.Tensor, lambda x: x.detach().cpu().numpy(), state_dict)

  bundle = StableHLOModelBundle(
      stablehlo_funcs=[StableHLOFunc(meta, stablehlo_content, stablehlo_text)],
      state_dict=exported_state_dict,
      additional_constants=additional_constants,
  )

  # Recover the global flag to not inline all scalars.
  torch_xla._XLAC._set_xla_all_numbers_special_scalars(False)
  # Recover the global XLA_HLO_DEBUG flag
  os.environ["XLA_HLO_DEBUG"] = xla_hlo_debug_env

  return bundle


def _save_program_bundle(
    bundle: StableHLOModelBundle,
    stablehlo_dir: os.PathLike,
    options: Optional[StableHLOExportOptions] = None) -> None:

  if options is None:
    options = StableHLOExportOptions()
  if options.save_weights:
    data_dir = os.path.join(stablehlo_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for key, val in bundle.state_dict.items():
      with open(os.path.join(stablehlo_dir, 'data', key), 'wb') as f:
        np.save(f, val)

  # save metadata and stablehlo bytecode
  func_dir = os.path.join(stablehlo_dir, 'functions')
  os.makedirs(func_dir, exist_ok=True)
  for func in bundle.stablehlo_funcs:
    with open(os.path.join(func_dir, func.meta.name + '.meta'), 'w') as f:
      json.dump(func.meta, f, cls=StableHLOJSONSerializer)
    with open(os.path.join(func_dir, func.meta.name + '.bytecode'), 'wb') as f:
      f.write(func.bytecode)
    if func.text is not None:
      with open(os.path.join(func_dir, func.meta.name + '.mlir'), 'w') as f:
        f.write(func.text)

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
  # if there is no weights/state_dict to be loaded, will not load weights
  if os.path.exists(os.path.join(stablehlo_dir, 'data')):
    for name, f in _iter_dir(os.path.join(stablehlo_dir, 'data')):
      state_dict[name] = np.load(f, allow_pickle=True)

  constants = []
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'constants')):
    # name of constants are ints
    constants.append((int(name), np.load(f, allow_pickle=True)))
  constants = [v for k, v in sorted(constants)]

  metas = []
  name_to_bytecode = {}
  name_to_text = {}
  stablehlo_funcs = []
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'functions')):
    if name.endswith('.meta'):
      metas.append(json.load(f, object_hook=stablehlo_obj_hook))
    elif name.endswith('.bytecode'):
      name_to_bytecode[os.path.splitext(name)[0]] = f.read()
    elif name.endswith('.mlir'):
      name_to_text[os.path.splitext(name)[0]] = f.read()

  for meta in metas:
    stablehlo_funcs.append(
        StableHLOFunc(meta, name_to_bytecode[meta.name],
                      name_to_text.get(meta.name)))

  return StableHLOModelBundle(
      stablehlo_funcs=stablehlo_funcs,
      additional_constants=constants,
      state_dict=state_dict)


def save_as_stablehlo(exported_model: 'ExportedProgram',
                      stablehlo_dir: os.PathLike,
                      options: Optional[StableHLOExportOptions] = None):
  """Convert a torch ExportedProgram to a callable backed by StableHLO, and save it to disk

  Args:
      exported_model: ExportedProgram - a pytorch ExportedProgram produced by torch.export.export
      stablehlo_dir: path to empty directory to create files.
      options: StableHLOExportOptions - options

  Files will contain stablehlo bytecode as well as tensor weights as numpy array.

  This files can be loaded into StableHLOGraphModule via StableHLOGraphModule.load

  Example:

  ```python
  model = ...
  exported = torch.export.export(model, args)
  save_as_stablehlo(exported, path)
  shlo_model = StableHLOGraphModule.load(path)
  assert shlo_model(*args) == model(*args)
  ```
  """

  if options is None:
    options = StableHLOExportOptions()
  shlo_program = exported_program_to_stablehlo(exported_model, options)
  shlo_program.save(stablehlo_dir, options)


def exported_program_to_stablehlo(
    exported_model: 'ExportedProgram',
    options: Optional[StableHLOExportOptions] = None) -> StableHLOGraphModule:
  """Convert a torch ExportedProgram to a callable backed by StableHLO.

  Args:
      model: ExportedProgram - a pytorch ExportedProgram produced by torch.export.export
      options: StableHLOExportOptions - options

  This function will return a callable backed by StableHLO such that,

  model(*sample_inputs) ==  export_torch_model(model, sample_inputs)(*sample_inputs)
  (up to numerics)

  In other words, returned callable have the same calling convention of the input model, and
  on the sample input, or inputs sufficiently similar* to sample input,
  it is will to return same result as the original model.

  * sufficiently similar input because this function will use tracing to extract the model operations
    so it might specialize on the shapes of the sample input.

  """
  if options is None:
    options = StableHLOExportOptions()
  bundle = _exported_program_to_stablehlo_bundle(exported_model, options)
  return StableHLOGraphModule(bundle)


def save_torch_model_as_stablehlo(
    torchmodel: torch.nn.Module,
    args: Tuple[Any],
    path: os.PathLike,
    options: Optional[StableHLOExportOptions] = None) -> None:
  """Convert a torch model to a callable backed by StableHLO.

  Args:
      model: torch.nn.Module - a pytorch model
      args: Tuple[torch.Tensor] - The input to this model
      path: path to empty directory to save the content
      options: StableHLOExportOptions


  This function will return a callable backed by StableHLO such that,

  model(*args) ==  export_torch_model(model, args)(*args)
  (up to numerics)

  In other words, returned callable have the same calling convention of the input model, and
  on the sample input, or inputs sufficiently similar* to sample input,
  it is will to return same result as the original model.

  * sufficiently similar input because this function will use tracing to extract the model operations
    so it might specialize on the shapes of the sample input.

  """
  exported = torch.export.export(torchmodel, args)
  if options is None:
    options = StableHLOExportOptions()
  options.override_tracing_arguments = args
  return save_as_stablehlo(exported, path, options)
