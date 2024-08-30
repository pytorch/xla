import copy
import dataclasses
import operator
import warnings

import functools
import itertools
import os
import time
from typing import Any, Dict, List, Set, Tuple, Union
from numbers import Number
from contextlib import contextmanager
from collections import deque

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.utils.fuser_utils import topo_sort

import torch._inductor
from torch._inductor.fx_passes.post_grad import ConstructorMoverPass

from torch.utils import _pytree as pytree
from torch_xla._dynamo import config

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as metrics
import torch_xla.core.xla_env_vars as xenv
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.utils.dlpack as torch_xla_dlpack

dynamo_debug = int(os.environ.get('XLA_DYNAMO_DEBUG', '0')) == 1
ptxla_debug = int(os.environ.get('PT_XLA_DEBUG', '0')) == 1


@contextmanager
def alias_with_buffer_donor_config(should_alias: bool = True):
  saved_config = torch_xla._XLAC._xla_get_should_alias_with_buffer_donor_config(
  )
  torch_xla._XLAC._xla_set_should_alias_with_buffer_donor_config(should_alias)
  try:
    yield saved_config
  finally:
    torch_xla._XLAC._xla_set_should_alias_with_buffer_donor_config(saved_config)


@dataclasses.dataclass
class GraphInputMatcher:
  """
  The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
  Specifically, those graph inputs corresponding to method parameters should be replaced with the
  arguments for the current call.

  Args:
    tensor_id_to_arg_idx: Dict[int, int] - Maps the tensor id to the HLO parameter index.
    graph_input_tensor_ids: List[int] - tensor_id for each TS/XLA graph input.
    graph_input_xla_values: List[torch.Tensor] - ivalue for each TS/XLA graph input.
                            Including both FX graph input tensors and weight tensors.
    xla_args_tensor_id: Set[int] - A set of tensor_ids for FX Graph inputs.
  """

  def __init__(self, tensor_id_to_arg_idx: Dict[int, int],
               graph_input_tensor_ids: List[int],
               graph_input_xla_values: List[torch.tensor],
               xla_args_tensor_id: Set[int]):
    self.tensor_id_to_arg_idx = tensor_id_to_arg_idx
    self.graph_input_tensor_ids = graph_input_tensor_ids
    # there are 2 categories of graph_input_tensors.
    # Category 1: those whose id are not found in tensor_id_to_arg_idx. These are
    # most likely const tensors and we can get its content from graph_input_tensors
    # Category 2: those whose id are found in tensor_id_to_arg_idx. We should get
    #  the tensor from method arguments.
    # For category 2, beause user inputs will be used for each run, we do not
    # cache those tensors in GraphInputMatcher.
    self.graph_input_xla_values = [
        None if tensor_id in xla_args_tensor_id else xla_value for tensor_id,
        xla_value in zip(graph_input_tensor_ids, graph_input_xla_values)
    ]
    self.seed_info_id = torch_xla._XLAC._get_seed_info_id()
    self.arg_idxs = []
    for tensor_id in self.graph_input_tensor_ids:
      self.arg_idxs.append(self.tensor_id_to_arg_idx.get(tensor_id, None))

  # get the real graph input tensors
  def __call__(self, args):
    real_input = [None] * len(self.graph_input_tensor_ids)
    current_index = 0
    for tensor_id, traced_xla_value, arg_idx in zip(self.graph_input_tensor_ids,
                                                    self.graph_input_xla_values,
                                                    self.arg_idxs):
      if arg_idx is None:
        # Instead of use trace time base seed, use the runtime
        # base seed here.
        if tensor_id == self.seed_info_id:
          str_device = str(traced_xla_value.device)
          inp = torch_xla._XLAC._get_base_seed_as_tensor(str_device)
          # update random seed here to avoid random operations always return
          # the same result. The seed update logic is the same as `mark_step` in
          # https://github.com/pytorch/pytorch/blob/6af6b8f728426fb7551630e28148c0017fa501bc/torch/csrc/lazy/core/lazy_graph_executor.cpp#L144C18-L144C51
          # Note: don't do `inp.item()` here since it will trigger a transferFromDevice
          xm.set_rng_state(
              (1012031 + torch_xla._XLAC._xla_get_rng_seed() * 7012063) %
              18446744073709551615, str_device)
        else:
          assert traced_xla_value is not None, "Traced Tensor cannot be None."
          inp = traced_xla_value
      else:
        assert traced_xla_value is None, "Graph input tensor should not be cached."
        inp = args[arg_idx]
      real_input[current_index] = inp
      current_index += 1
    assert current_index == len(self.graph_input_tensor_ids)
    return real_input


def get_fallback_ops():
  return metrics.executed_fallback_ops()


# Checks that all input args that are tensors are on the same device.
def _get_input_arg_device(input_args: tuple) -> torch.device:
  device = None
  for arg in input_args:
    if not isinstance(arg, torch.Tensor):
      continue

    if device is None:
      device = arg.device
    else:
      assert arg.device == device, "Not all args are on the same device."

    return device


# Returns True if all the input args are on a CUDA device.
def _args_on_cuda(input_args: tuple) -> bool:
  input_device: torch.device = _get_input_arg_device(input_args)
  if input_device is None:
    return False

  return input_device.type == "cuda"


# Given an input list, moves the tensors to the given target_device.
# The output order will be the same as the input. Non tensors will also still
# be in the list.
def _maybe_move_tensors_to_device(tensors: tuple,
                                  target_device: torch.device) -> tuple:
  assert target_device, "Moving tensors to None device not supported"

  moved_tensors = []
  for tensor in tensors:
    if not isinstance(tensor, torch.Tensor):
      moved_tensors.append(tensor)
      continue

    if tensor.device == target_device:
      moved_tensors.append(tensor)
      continue

    if dynamo_debug:
      print("Moving Tensor {} to device {}".format(tensor, target_device))

    zero_copy_enabled = xu.getenv_as(xenv.ZERO_COPY_ENABLED, bool, defval=False)
    if zero_copy_enabled and tensor.device.type == 'cuda' and target_device.type == 'xla':
      # If the input cuda tensor requires gradient, we need to call detach. Otherwise, we'd get the error "RuntimeError: Can't export tensors that require gradient, use tensor.detach()"
      moved_tensor = torch_xla_dlpack.from_dlpack(tensor.detach())
    elif zero_copy_enabled and tensor.device.type == 'xla' and target_device.type == 'cuda':
      # mark_step is need to make sure the pjrt buffer is valid.
      xm.mark_step()
      moved_tensor = torch_xla_dlpack.from_xla_cuda_to_cuda(tensor)
    else:
      # Have to move to CPU before moving it to target device.
      cpu_device: torch.device = torch.device("cpu")
      moved_tensor = tensor.to(cpu_device)
      moved_tensor = moved_tensor.to(target_device)

    # Explicitly have to copy requires_grad attribute because it's dropped
    # with torch.to(..)
    moved_tensor.requires_grad = tensor.requires_grad
    moved_tensors.append(moved_tensor)

  return tuple(moved_tensors)


def _split_xla_args_tensor_sym_constant(args):
  tensors = deque(maxlen=len(args))
  constants = []
  for arg in args:
    if isinstance(arg, torch.Tensor):
      tensors.append(arg)
    elif isinstance(arg, Number):
      constants.append(arg)
    else:
      assert False, "argument to the xla dynamo bridge should be either a number or tensor"

  return tensors, tuple(constants)


class Deduper:

  def __init__(self):
    # origlist index to dedupedlist index
    self.permute_for_orig = None

  def dedup(self, origlist):
    self.permute_for_orig = []
    deduped_ids = dict()
    deduped_list = []
    for item in origlist:
      if isinstance(item, torch.Tensor):
        item_id = torch_xla._XLAC._unique_id_for_ir_and_data(item)
      else:
        item_id = id(item)
      if item_id not in deduped_ids:
        deduped_ids[item_id] = len(deduped_ids)
        deduped_list.append(item)
      self.permute_for_orig.append(deduped_ids[item_id])

    return deduped_list

  def recover(self, deduped_list):
    assert len(self.permute_for_orig) >= len(deduped_list)
    return [deduped_list[i] for i in self.permute_for_orig]


class SpecialReturnHandler:
  """
  In this class we handle 2 types of special outputs
  1. dump return
  We Define dumb return as an output that is also an input.
  Torch xla does not return such tensors as its graph output. That breaks the
  API contract with the caller of the graph. Also AOTAutograd
  may generate such a graph quite often.

  To avoid break the contract with the user of the GraphModule, we need
  add those outputs manually.

  Check https://github.com/pytorch/pytorch/pull/89536 for details.

  AOTAutograd may also generate graph with duplicated return item.
  E.g. https://gist.github.com/shunting314/e60df8ac21fbe2494337c10d02bd78dc
  (this is a graph generated for a model with a single BatchNorm2d)
  XLA will dedup those duplicate items, but we need recover the duplications to maintain
  the contract with the caller.

  2. constant output from dynamic compile
  In the case of the `torch.compile(Dynamic=True)` there might be some int or float outputs related
  to the dynmiac dimension of the tensor. These numbers are static for a given input shape
  combinations so we can cache them and inject to the final result directly.
  """

  def __init__(self, trace_inputs, trace_outputs,
               trace_inputs_inplace_update_bool, constant_outputs_and_indexes):
    self.trace_inputs = trace_inputs
    self.trace_outputs = trace_outputs
    self.constant_outputs_and_indexes = constant_outputs_and_indexes

    # dedup the traced outputs first
    self.deduper = Deduper()
    self.deduped_trace_outputs = self.deduper.dedup(self.trace_outputs)

    # record the output that is also a input
    trace_inputs_id2pos = {
        id(x): pos for pos, x in enumerate(self.trace_inputs)
    }
    self.trace_outputs_pos_to_inputs_pos = []
    for out_pos, out in enumerate(self.deduped_trace_outputs):
      in_pos = trace_inputs_id2pos.get(id(out), None)
      if in_pos is not None and not trace_inputs_inplace_update_bool[in_pos]:
        self.trace_outputs_pos_to_inputs_pos.append((out_pos, in_pos))

  def addDumbReturn(self, real_inputs, real_outputs):
    for out_pos, in_pos in self.trace_outputs_pos_to_inputs_pos:
      assert in_pos < len(real_inputs)
      # equals is fine since we can append an item at the end
      assert out_pos <= len(real_outputs)

      real_outputs.insert(out_pos, real_inputs[in_pos])

    ret = self.deduper.recover(real_outputs)

    if len(self.constant_outputs_and_indexes) != 0:
      # insert the int outputs back to the res
      for index, value in self.constant_outputs_and_indexes:
        ret.insert(index, value)

    return ret


class NoneRemover:
  """
  torchxla pybind APIs that accepts a Tensor list does not expect None value on
  the list. But some graph (e.g. backward graph generated by aot autograd) may
  return a None value. We need strip those None value before sending the list to
  those torchxla APIs. We need add None value back later after running the
  compiled graph from torchxla.
  """

  def __init__(self):
    self.none_poslist = []

  def remove_nones(self, value_list):
    """
    Remove none from value_list. value_list will be inplace updated.
    The original position of None values are recorded.
    """
    num = len(value_list)

    # work in reverse order
    for i in reversed(range(num)):
      if value_list[i] is None:
        self.none_poslist.append(i)
        del value_list[i]

    self.none_poslist.reverse()

  def add_nones(self, value_list):
    """
    Add nones to value_list according to self.none_poslist. value_list
    is inplace updated.
    """
    for pos in self.none_poslist:
      value_list.insert(pos, None)


def is_xla_tensor(tensor: torch.Tensor) -> bool:
  return tensor.device.type == "xla"


def extract_graph_helper(xla_model: torch.fx.GraphModule,
                         sym_constants_to_graph_vars: Dict[Tuple[Union[int,
                                                                       float],
                                                                 ...],
                                                           Tuple[Any, ...]]):
  # Don't reset the scope as we might be under some profiler trace scope.
  xm.mark_step(reset_scope=False)
  # FX Graph inputs passed from Dynamo. xla_args are XLA Tensors.
  xla_args = xla_model.xla_args
  # check [Note: Dynamo real-time input-shape cache look-up]
  # xla_arg might contain symint and we want to filter it out in some use cases.
  # We want to use `xla_args` instead of `xla_args_tensor_only` only when we do
  # `res = xla_model(xla_args)` since `xla_models` expects sym constants as inputs
  xla_args_tensor_only, sym_constants = _split_xla_args_tensor_sym_constant(
      xla_args)
  xla_args_tensor_ids = set(
      pytree.tree_map_only(
          torch.Tensor,
          lambda xla_arg: torch_xla._XLAC._xla_get_tensor_id(xla_arg),
          xla_args_tensor_only))
  assert all(
      map(
          is_xla_tensor,
          filter(
              lambda x: isinstance(x, torch.Tensor),
              itertools.chain(xla_model.parameters(), xla_args),
          ),
      )), "All tensors should be on xla"

  # Clone the input tensors which can be used to restore the origional value of the xla_args
  # if model applied inplace operations to the input.
  # Note that we currently can't use `cloned_xla_args` to perform the tracing since cloned
  # Tensor and base tensor will share the same XLAData and XLAData only store the tensor ID
  # of the first XLATensor, which is the base tensor. When we try to map the XLAData back to
  # TensorID in `_get_tensors_xla_device_data_node` to create the mapping, the wrong Tensor ID
  # will be returned.
  # TODO(JackCaoG): fix the cloned tensor can't be used to warm up the cache.
  cloned_xla_args = [torch.clone(xla_arg) for xla_arg in xla_args_tensor_only]

  index_and_xla_tensor_args = [
      (i, xla_arg) for i, xla_arg in enumerate(xla_args_tensor_only)
  ]

  index_and_tensor_ids = [(index, torch_xla._XLAC._xla_get_tensor_id(xla_arg))
                          for index, xla_arg in index_and_xla_tensor_args]

  if dynamo_debug:
    print(f"Graph Module:\n{xla_model.code}")

  tensor_id_to_arg_idx = {
      tensor_id: index for index, tensor_id in index_and_tensor_ids
  }

  if xr.is_spmd():
    xla_args_sharding_spec = torch_xla._XLAC._get_xla_sharding_specs(
        xla_args_tensor_only)
  else:
    xla_args_sharding_spec = ()

  # To trace the model we need `xla_args` instead of `xla_args_tensor_only`
  xla_out = xla_model(*xla_args)
  if not isinstance(xla_out, (tuple, list)):
    xla_out = (xla_out,)

  none_remover = NoneRemover()
  none_remover.remove_nones(xla_out)

  xla_out_ids = {id(x) for x in xla_out}

  # If an arg is being in place updated by model, we need to include arg as part of the graph result.
  xla_args_need_update_bool = torch_xla._XLAC._check_tensor_need_materialization(
      [tensor for _, tensor in index_and_xla_tensor_args])
  xla_args_need_update = []
  arg_index_to_need_update_index = {}
  for i, need_update in enumerate(xla_args_need_update_bool):
    # Don't add inplace updated argument to the list if it's already
    # being returned
    index, tensor = index_and_xla_tensor_args[i]
    if need_update and id(tensor) not in xla_out_ids:
      arg_index_to_need_update_index[index] = len(xla_args_need_update)
      xla_args_need_update.append(tensor)

  args_and_out = tuple(xla_args_need_update) + tuple(xla_out)
  # args_and_out should be tensor only, in the dynamic cases there might be
  # symint or symfloat return as the result. In that case we want to extract them and separate
  # them from the device computation.
  constant_outputs_and_indexes = []
  args_and_out_tensor_only = []
  for i in range(len(args_and_out)):
    arg = args_and_out[i]
    if not isinstance(arg, torch.Tensor):
      assert isinstance(arg, Number)
      constant_outputs_and_indexes.append((i, arg))
    else:
      args_and_out_tensor_only.append(arg)

  special_return_handler = SpecialReturnHandler(xla_args_tensor_only,
                                                args_and_out_tensor_only,
                                                xla_args_need_update_bool,
                                                constant_outputs_and_indexes)

  # There is a `mark_step` in the beginning of this function call, we need to wait
  # for that to finish before retriving the device data nodes.
  xm.wait_device_ops()
  # Collect all device data nodes that is needed to compute the args_and_out_tensor_only
  # and wrap those device data nodes inside a at::tensor(graph_input_xla_values).
  # Return the tensor_id that is corresponding to every device_data node as
  # graph_input_tensor_ids.
  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(
      args_and_out_tensor_only)
  assert len(graph_input_tensor_ids) == len(
      graph_input_xla_values
  ), f"{len(graph_input_tensor_ids)} v.s. {len(graph_input_xla_values)}"
  graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx,
                                          graph_input_tensor_ids,
                                          graph_input_xla_values,
                                          xla_args_tensor_ids)

  if dynamo_debug:
    print(f"Number of HLO Input: {len(graph_input_tensor_ids)}")
    buffer_donor_count = 0
    for graph_input in graph_input_xla_values:
      buffer_donor_count += 1 if isinstance(
          graph_input, torch.Tensor) and torch_xla._XLAC._get_buffer_donation(
              graph_input) else 0
    print(f"Number of HLO Output: {len(args_and_out_tensor_only)}")
    print(
        f"Number of HLO Input can be aliased with Output: {buffer_donor_count}")
    print(
        f"XLA IR Text: \n{torch_xla._XLAC._get_xla_tensors_text(args_and_out_tensor_only)}"
    )

  with alias_with_buffer_donor_config() as saved_config:
    # calculate graph hash
    graph_hash = torch_xla._XLAC._get_graph_hash(args_and_out_tensor_only)
    if dynamo_debug:
      print("Graph Hash: ", graph_hash)
    # compiles and cache graph rooted at tensors in 'args_and_out_tensor_only'
    torch_xla._XLAC._xla_warm_up_cache(args_and_out_tensor_only, [])

  # Restore the origional `xla_args`. Dynamo passed the real tensor as
  # `xla_args`` and we performend the tracing on them. During the tracing,
  # in place update will replace the underlying DeviceData of the `xla_args`.
  # Note that this needs to happens before `_clear_pending_irs` otherwise
  # the additional IR generated by `copy_` won't be cleared.
  assert len(xla_args_need_update_bool) == len(xla_args_tensor_only)
  for i, need_update in enumerate(xla_args_need_update_bool):
    if need_update and isinstance(xla_args[i], torch.Tensor):
      xla_args_tensor_only[i].copy_(cloned_xla_args[i])

  # Remove all of the pending IR from all live tensors. The assumptions are
  # 1. With the  `xm.mark_step` in the beginning of this call, every XLATensor
  # should be materialized
  # 2. All of the pending IRs are result of our warm up cache tracing and they
  # should be removed to avoid extra computation executed and in place updates op
  # mistakenlly update the input tensors.
  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))

  vars_to_return = (xla_args_sharding_spec, args_and_out, graph_hash,
                    arg_index_to_need_update_index, none_remover,
                    graph_input_matcher, special_return_handler,
                    xla_args_need_update)
  # populate the cache
  sym_constants_to_graph_vars[sym_constants] = vars_to_return

  return vars_to_return


def extract_internal(xla_model: torch.fx.GraphModule):
  if dynamo_debug:
    print(
        '\n=================== OpenXLA Dynamo Compile Debug Begin ==================='
    )
    print('Input Tensor Debug Infos:')
    for xla_arg in xla_model.xla_args:
      if isinstance(xla_arg, torch.Tensor):
        print(torch_xla._XLAC._get_xla_tensor_debug_info(xla_arg))

  # [Note: Dynamo real-time input-shape cache look-up]
  # We maintain a mapping of sym ints from arg to outputs of extract_graph_helper.
  # When dynamic=True in torch.compile call or `mark_dynamic` being called for one
  # of the input to the compiled function, TorchDynamo will not trigger a
  # new recompile. Then TorchXLA needs to figure out if these input shapes have
  # been seen before.
  # Keys: tuple of sym constants, most likely either a python int or a python float
  # Values: tuple of (xla_args_sharding_spec, args_and_out, graph_hash,
  # arg_index_to_need_update_index, none_remover, graph_input_matcher,
  # special_return_handler, xla_args_need_update).
  sym_constants_to_graph_vars: Dict[Tuple[Union[int, float], ...],
                                    Tuple[Any, ...]] = {}

  (xla_args_sharding_spec, args_and_out, graph_hash,
   arg_index_to_need_update_index, none_remover, graph_input_matcher,
   special_return_handler,
   xla_args_need_update) = extract_graph_helper(xla_model,
                                                sym_constants_to_graph_vars)
  skip_checking_input_sharding_threashold = xu.getenv_as(
      'XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD', int, 5)

  def optimized_mod(*args: tuple):
    nonlocal xla_model
    nonlocal skip_checking_input_sharding_threashold
    nonlocal sym_constants_to_graph_vars

    # See [Note: Dynamo real-time input-shape cache look-up] above.
    xla_args_tensor_only, sym_constants = _split_xla_args_tensor_sym_constant(
        args)
    if sym_constants in sym_constants_to_graph_vars:
      (xla_args_sharding_spec, args_and_out, graph_hash,
       arg_index_to_need_update_index, none_remover, graph_input_matcher,
       special_return_handler,
       xla_args_need_update) = sym_constants_to_graph_vars[sym_constants]
    else:
      xla_model.xla_args = args
      (xla_args_sharding_spec, args_and_out, graph_hash,
       arg_index_to_need_update_index, none_remover, graph_input_matcher,
       special_return_handler, xla_args_need_update) = extract_graph_helper(
           xla_model, sym_constants_to_graph_vars)

    original_device: torch.device = _get_input_arg_device(args)
    is_cuda_args: bool = False
    if original_device:
      is_cuda_args = original_device.type == "cuda"

    if is_cuda_args:
      args = _maybe_move_tensors_to_device(args, xm.xla_device())

    if not config.skip_input_data_check:
      # mark_step needs to be blocking since we want to access args's XLADatas
      # and they can't be placeholder.
      input_tensors_to_sync = [
          xla_args_tensor_only[i] for i, x in enumerate(
              torch_xla._XLAC._check_tensor_need_materialization(
                  xla_args_tensor_only)) if x
      ]

      if len(input_tensors_to_sync) > 0:
        torch_xla._XLAC._xla_increment_counter('DynamoSyncInputExecuteTime', 1)
        torch_xla._XLAC._xla_sync_multi(
            input_tensors_to_sync, devices=[], wait=True, sync_xla_data=True)

    # If input sharding has changed from the previous program, dynamo current can
    # not detect this. It will mistakenly believe the program is the same. We need
    # to retrace it here.
    if xr.is_spmd():
      # if the input sharding was the same for skip_checking_input_sharding_threashold times
      # we will skip checking the input sharding since it can be expensive.
      if skip_checking_input_sharding_threashold > 0:
        if torch_xla._XLAC._get_xla_sharding_specs(
            args) != xla_args_sharding_spec:
          # update the xla_args with the input with new sharding and retrace
          xla_model.xla_args = args
          (xla_args_sharding_spec, args_and_out_copy, graph_hash,
           arg_index_to_need_update_index, none_remover, graph_input_matcher,
           special_return_handler, xla_args_need_update) = extract_graph_helper(
               xla_model, sym_constants_to_graph_vars)
          skip_checking_input_sharding_threashold = xu.getenv_as(
              'XLA_DYNAMO_INPUT_SHARDING_CHECK_THRESHOLD', int, 5)
        else:
          skip_checking_input_sharding_threashold -= 1

    if len(args_and_out) == 0:
      return ()

    # graph input should be tensor only
    graph_input = graph_input_matcher(xla_args_tensor_only)
    res = torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)
    res = special_return_handler.addDumbReturn(xla_args_tensor_only, res)

    assert len(res) == len(args_and_out), f"{len(res)} v.s. {len(args_and_out)}"
    ncopy = 0

    for arg_index, res_index in arg_index_to_need_update_index.items():
      args[arg_index].copy_(res[res_index])

    # First few elements might be xla_args that needs to be in place updated
    result = res[len(xla_args_need_update):]

    none_remover.add_nones(result)
    if is_cuda_args:
      result = _maybe_move_tensors_to_device(tuple(result), original_device)

    if len(result) == 1:
      return result[0]
    else:
      return result

  if dynamo_debug:
    print(
        '=================== OpenXLA Dynamo Compile Debug End =====================\n'
    )
  return optimized_mod


class UnsupportedNodesCollector(torch.fx.Interpreter):

  def __init__(self, module):
    super().__init__(module)
    self._unsupported_nodes = []

  def run_node(self, n: torch.fx.Node):
    # We need to restore this metric count later, so save it in a separate variable
    dynamo_extract_graph_helper_metric_count = metrics.counter_value(
        'DynamoExtractCompiledGraph')

    metrics.clear_counters()
    result = super().run_node(n)
    fallback_ops = get_fallback_ops()
    if len(fallback_ops) > 0:
      self._unsupported_nodes.append(n)
    else:
      # Check whether the tensors contained in value are all XLA tensors.
      def all_tensors_on_xla_device(value):
        if isinstance(value, torch.Tensor):
          return is_xla_tensor(value)
        if isinstance(value, (list, tuple)):
          return all(all_tensors_on_xla_device(v) for v in value)
        # Not a tensor nor a container.
        return True

      # Check whether the current node is supported or not.
      #
      # A supported node has the following characteristics:
      # - a node whose result is a composition of XLA tensors:
      #   avoids non-XLA tensors as FX graph return value.
      result_is_supported = all_tensors_on_xla_device(result)

      # - a node that whose tensor arguments are XLA tensors:
      #   avoids non-XLA tensors as FX graph arguments.
      args, kwargs = self.fetch_args_kwargs_from_env(n)
      args_are_supported = all(
          all_tensors_on_xla_device(v)
          for v in itertools.chain(args, kwargs.values()))

      # If the current node is NOT supported, we add it to
      # the _unsupported_nodes list.
      if not (result_is_supported and args_are_supported):
        self._unsupported_nodes.append(n)

    # Restore this metric counter
    torch_xla._XLAC._xla_increment_counter(
        'DynamoExtractCompiledGraph', dynamo_extract_graph_helper_metric_count)

    return result

  def get_unsupported_nodes(self):
    return self._unsupported_nodes


class InputCollector(torch.fx.Interpreter):

  def __init__(self, module):
    super().__init__(module)

  def call_module(self, target, args, kwargs):
    if "fused_" in target:
      submod = self.fetch_attr(target)
      submod.xla_args = args
    return super().call_module(target, args, kwargs)


class XLAConstructorMoverPass(ConstructorMoverPass):

  def __init__(self):
    super().__init__("xla")

  # Returns whether node may take CPU tensors as argument, while
  # returning an XLA tensor. For example, index.Tensor may take CPU
  # indices, while its return value remains on XLA.
  def allow_cpu_device(self, node: torch.fx.Node):
    if super().allow_cpu_device(node):
      return True

    # also allow _to_copy calls.
    # this is a recurring pattern when creating new tensors.
    if node.target != torch.ops.aten._to_copy.default:
      return False

    # in this case, there should be a device keyword-argument
    # where its type is the same as the target device.
    device = node.kwargs.get("device")
    return (device is not None and device.type == self.target)


def extract_compiled_graph(xla_model: torch.fx.GraphModule, xla_args):
  torch_xla._XLAC._xla_increment_counter('DynamoExtractCompiledGraph', 1)

  with torch_xla.experimental.eager_mode_context(False):
    return extract_compiled_graph_helper(xla_model, xla_args)


def _clear_pending_irs_on_args(args_tensor_only, cloned_args):
  # if args_tensor_only has pending IR which means there is a in place operations
  # happened. We don't want to execute that operation yet, so we will replace the
  # pending IR with the cloned arg.
  args_need_update_bool = torch_xla._XLAC._check_tensor_need_materialization(
      args_tensor_only)

  for i, need_update in enumerate(args_need_update_bool):
    if need_update and isinstance(args_tensor_only[i], torch.Tensor):
      args_tensor_only[i].copy_(cloned_args[i])


def partition_fx_graph_for_cpu_fallback(xla_model, xla_args, all_xla_args,
                                        all_xla_args_tensor_only):
  # below logic will try to partition the fx graph based on the fallback ops.
  cloned_args = [
      torch.clone(xla_arg) if isinstance(xla_arg, torch.Tensor) else xla_arg
      for xla_arg in all_xla_args
  ]

  # execute model once to collect fallback ops
  collector = UnsupportedNodesCollector(xla_model)
  collector.run(*xla_args)
  unsupported_nodes = collector.get_unsupported_nodes()
  if (ptxla_debug or dynamo_debug) and len(unsupported_nodes) > 0:
    print('Dynamo fallback ops are' + str(unsupported_nodes) +
          '. Please open a GitHub issue with the above op lowering requests.')

  # UnsupportedNodesCollector might trigger in place ops, need to clear them here.
  _clear_pending_irs_on_args(all_xla_args_tensor_only, cloned_args)

  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))

  class XlaOperatorSupport(torch.fx.passes.operator_support.OperatorSupport):

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
      return node.op in [
          "call_function", "call_module", "call_method"
      ] and (node not in unsupported_nodes or node.target == operator.getitem)

  # partition the model
  supported_ops = XlaOperatorSupport()
  partitioner = CapabilityBasedPartitioner(
      xla_model, supported_ops, allows_single_node_partition=True)
  partitions = partitioner.propose_partitions()

  # propose_partitions() does not guarantee topolgical order, so sort it manually
  for partition in partitions:
    partition.nodes = topo_sort(partition.nodes)

  # fuse partitions and exectue to collect inputs
  partitioned_graph = partitioner.fuse_partitions(partitions)
  InputCollector(partitioned_graph).run(*xla_args)

  # InputCollector might trigger in place ops, need to clear them here.
  _clear_pending_irs_on_args(all_xla_args_tensor_only, cloned_args)

  # compile each submodule and replace it with a call
  for node in partitioned_graph.graph.nodes:
    if node.op == "call_module" and "fused_" in node.name:
      fused_module = getattr(partitioned_graph, node.name)
      partitioned_graph.delete_submodule(node.target)
      with partitioned_graph.graph.inserting_after(node):
        new_node = partitioned_graph.graph.call_function(
            extract_internal(fused_module), node.args, None)
        node.replace_all_uses_with(new_node)
      partitioned_graph.graph.erase_node(node)

  partitioned_graph.recompile()

  return partitioned_graph


def extract_compiled_graph_helper(xla_model: torch.fx.GraphModule, xla_args):
  if _args_on_cuda(xla_args):
    xla_args = tuple(_maybe_move_tensors_to_device(xla_args, xm.xla_device()))

  # Synchronize xla_args, so that each FunctionalTensorWrapper argument updates its
  # value reference before actually computing it.
  for a in xla_args:
    if isinstance(a, torch.Tensor) and torch._is_functional_tensor(a):
      torch._functionalize_sync(a)

  # This call is critical to make sure xla_args' tensor id show up in graph_input_tensor_ids.
  # Don't reset the scope as we might be under some profiler trace scope.
  xm.mark_step(reset_scope=False)

  # Find tensor constructor nodes that create CPU tensors, and make
  # them create XLA tensors, where possible, instead. i.e. replace the
  # device=CPU keyword-argument of tensor constructor nodes by XLA.
  XLAConstructorMoverPass()(xla_model.graph)

  # If a model's `forward` function has an in-place op that acts on its `self.tensor`, the
  # `self.tensor` is not included as a part of the `xla_args` and does not get materialized.
  # This explicitly fetches the `self.tensor`s if they exist.
  self_args = []
  for name, buffer in xla_model.named_buffers():
    if "self" in name:
      self_args.append(buffer)

  all_xla_args = list(xla_args) + self_args
  all_xla_args_tensor_only = [
      xla_arg for xla_arg in all_xla_args if isinstance(xla_arg, torch.Tensor)
  ]

  for xla_arg in xla_args:
    if isinstance(xla_arg, torch.Tensor) and xla_arg.device.type != 'xla':
      warnings.warn(
          "Found tensor with shape " + str(xla_arg.size()) + " on " +
          str(xla_arg.device) +
          ". Please move all tensors to xla device to execute on XLA device.")

  # when there are symints in the `xla_args`, don't run partitioner since it will
  # separate the symints in to a spearate graph and mess up the caching.
  if len(all_xla_args) != len(all_xla_args_tensor_only):
    xla_model.xla_args = xla_args
    return extract_internal(xla_model)
  else:
    return partition_fx_graph_for_cpu_fallback(xla_model, xla_args,
                                               all_xla_args,
                                               all_xla_args_tensor_only)
