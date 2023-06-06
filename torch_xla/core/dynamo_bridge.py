import copy
import dataclasses
import operator

import functools
import itertools
import os
import time
from typing import Any, Dict, List

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as metrics

debug = os.environ.get("TORCH_XLA_DEBUG") == "1"


@dataclasses.dataclass
class GraphInputMatcher:
  """
  The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
  Specifically, those graph inputs corresponding to method parameters should be replaced with the
  arguments for the current call.

  tensor_id_to_arg_idx maps the tensor id to the parameter index.
  graph_input_tensor_ids, graph_input_xla_values list the tensor_id and ivalue for each of the
  TS/XLA graph inputs.
  """

  tensor_id_to_arg_idx: Dict[int, int]
  graph_input_tensor_ids: List[int]
  # there are 2 categories of graph_input_tensors.
  # Category 1: those whose id are not found in tensor_id_to_arg_idx. These are
  # most likely const tensors and we can get its content from graph_input_tensors
  # Category 2: those whose id are found in tensor_id_to_arg_idx. We should get
  #  the tensor from method arguments
  graph_input_xla_values: List[Any]

  # get the real graph input tensors
  def __call__(self, args):
    real_input = []
    for tensor_id, traced_xla_value in zip(self.graph_input_tensor_ids,
                                           self.graph_input_xla_values):
      arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
      # Instead of use trace time base seed, use the runtime
      # base seed here.
      if tensor_id == torch_xla._XLAC._get_seed_info_id():
        inp = torch_xla._XLAC._get_base_seed_as_tensor(
            str(traced_xla_value.device))
      elif arg_idx is None:
        inp = traced_xla_value
      else:
        inp = args[arg_idx]
      real_input.append(inp)
    return real_input


def get_fallback_ops():
  fallback_ops = []
  for opname in metrics.counter_names():
    if "aten::" not in opname:
      continue
    val = int(metrics.counter_value(opname))
    if val > 0:
      fallback_ops.append(f"{opname}={val}")

  return fallback_ops


class Deduper:

  def __init__(self):
    # origlist index to dedupedlist index
    self.permute_for_orig = None

  def dedup(self, origlist):
    self.permute_for_orig = []
    deduped_ids = dict()
    deduped_list = []
    for item in origlist:
      item_id = id(item)
      if item_id not in deduped_ids:
        deduped_ids[item_id] = len(deduped_ids)
        deduped_list.append(item)
      self.permute_for_orig.append(deduped_ids[item_id])

    return deduped_list

  def recover(self, deduped_list):
    assert len(self.permute_for_orig) >= len(deduped_list)
    return [deduped_list[i] for i in self.permute_for_orig]


class DumbReturnHandler:
  """
  Define dumb return as an output that is also an input.
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
  """

  def __init__(self, trace_inputs, trace_outputs,
               trace_inputs_inplace_update_bool):
    self.trace_inputs = trace_inputs
    self.trace_outputs = trace_outputs

    # dedup the traced outputs first
    self.deduper = Deduper()
    self.deduped_trace_outputs = self.deduper.dedup(self.trace_outputs)

    if debug:
      print(
          f"Number of duplicated outputs {len(self.trace_outputs) - len(self.deduped_trace_outputs)})"
      )

    # record the output that is also a input
    trace_inputs_id2pos = {
        id(x): pos for pos, x in enumerate(self.trace_inputs)
    }
    self.trace_outputs_pos_to_inputs_pos = []
    for out_pos, out in enumerate(self.deduped_trace_outputs):
      in_pos = trace_inputs_id2pos.get(id(out), None)
      if in_pos is not None and not trace_inputs_inplace_update_bool[in_pos]:
        self.trace_outputs_pos_to_inputs_pos.append((out_pos, in_pos))

    if debug:
      print(
          f"Number trace input {len(trace_inputs)}, number trace output {len(trace_outputs)}"
      )
      print(
          f"Found {len(self.trace_outputs_pos_to_inputs_pos)} dumb returns: {self.trace_outputs_pos_to_inputs_pos}"
      )

  def addDumbReturn(self, real_inputs, real_outputs):
    for out_pos, in_pos in self.trace_outputs_pos_to_inputs_pos:
      assert in_pos < len(real_inputs)
      # equals is fine since we can append an item at the end
      assert out_pos <= len(real_outputs)

      real_outputs.insert(out_pos, real_inputs[in_pos])

    ret = self.deduper.recover(real_outputs)
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


def extract_internal(xla_model: torch.fx.GraphModule):
  xla_args = xla_model.xla_args
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
  cloned_xla_args = [
      torch.clone(xla_arg) if isinstance(xla_arg, torch.Tensor) else xla_arg
      for xla_arg in xla_args
  ]

  args_tensor_ids = [
      torch_xla._XLAC._xla_get_tensor_id(xla_arg) for xla_arg in xla_args
  ]

  if debug:
    print(f"Graph module:\n{xla_model.code}")
    print(f"args_tensor_ids {args_tensor_ids}")

  tensor_id_to_arg_idx = {
      tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)
  }

  xla_out = xla_model(*xla_args)
  if not isinstance(xla_out, (tuple, list)):
    xla_out = (xla_out,)

  none_remover = NoneRemover()
  none_remover.remove_nones(xla_out)

  xla_out_ids = {id(x) for x in xla_out}

  # If a arg is being in place updated by model, we need to include arg as part of the graph result.
  xla_args_need_update_bool = torch_xla._XLAC._check_tensor_need_materialization(
      xla_args)
  xla_args_need_update = []
  arg_index_to_need_update_index = {}
  for i, need_update in enumerate(xla_args_need_update_bool):
    # Don't add inplace updated argument to the list if it's already
    # being returned
    if need_update and id(xla_args[i]) not in xla_out_ids:
      arg_index_to_need_update_index[i] = len(xla_args_need_update)
      xla_args_need_update.append(xla_args[i])

  args_and_out = tuple(xla_args_need_update) + tuple(xla_out)

  if debug:
    print(f"#inplace update: {len(xla_args_need_update)}")
    print(f"XLA IR Text: {torch_xla._XLAC._get_xla_tensors_text(args_and_out)}")

  # calculate graph hash
  dumb_return_handler = DumbReturnHandler(xla_args, args_and_out,
                                          xla_args_need_update_bool)
  graph_hash = torch_xla._XLAC._get_graph_hash(args_and_out)
  if debug:
    print("graph_hash", graph_hash)

  # Collect all device data nodes that is needed to compute the args_and_out
  # and wrap those device data nodes inside a at::tensor(graph_input_xla_values).
  # Return the tensor_id that is corresponding to every device_data node as
  # graph_input_tensor_ids.
  (
      graph_input_tensor_ids,
      graph_input_xla_values,
  ) = torch_xla._XLAC._get_tensors_xla_device_data_node(args_and_out)
  if debug:
    print(f"graph_input_tensor_ids {graph_input_tensor_ids}")
  assert len(graph_input_tensor_ids) == len(
      graph_input_xla_values
  ), f"{len(graph_input_tensor_ids)} v.s. {len(graph_input_xla_values)}"
  graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx,
                                          graph_input_tensor_ids,
                                          graph_input_xla_values)

  # compiles and cache graph rooted at tensors in 'args_and_out'
  torch_xla._XLAC._xla_warm_up_cache(args_and_out, [])

  # Restore the origional `xla_args`. Dynamo passed the real tensor as
  # `xla_args`` and we performend the tracing on them. During the tracing,
  # in place update will replace the underlying DeviceData of the `xla_args`.
  # Note that this needs to happens before `_clear_pending_irs` otherwise
  # the additional IR generated by `copy_` won't be cleared.
  if xla_args_need_update_bool:
    for xla_arg, cloned_xla_arg in zip(xla_args, cloned_xla_args):
      if isinstance(xla_arg, torch.Tensor):
        xla_arg.copy_(cloned_xla_arg)

  # Remove all of the pending IR from all live tensors. The assumptions are
  # 1. With the  `xm.mark_step` in the beginning of this call, every XLATensor
  # should be materialized
  # 2. All of the pending IRs are result of our warm up cache tracing and they
  # should be removed to avoid extra computation executed and in place updates op
  # mistakenlly update the input tensors.
  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))

  def optimized_mod(*args):
    # mark_step needs to be blocking since we want to access args's XLADatas
    # and they can't be placeholder.
    if any(torch_xla._XLAC._check_tensor_need_materialization(args)):
      xm.mark_step(wait=True)
    enter_ts = time.time()
    if len(args_and_out) == 0:
      return ()

    assert len(args) > 0  # can not handle no args case for now
    graph_input = graph_input_matcher(args)
    start_ts = time.time()
    res = torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)
    res = dumb_return_handler.addDumbReturn(args, res)
    if debug:
      print(
          f"torchxla reuse compiled graph run_cached_graph takes {time.time() - start_ts} seconds"
      )

    args_inplace_update_ts = time.time()
    assert len(res) == len(args_and_out), f"{len(res)} v.s. {len(args_and_out)}"
    ncopy = 0

    for arg_index, res_index in arg_index_to_need_update_index.items():
      args[arg_index].copy_(res[res_index])

    if debug:
      print(
          f"Copy {ncopy} args takes {time.time() - args_inplace_update_ts} seconds"
      )

    # First few elements might be xla_args that needs to be in place updated
    result = res[len(xla_args_need_update):]
    if debug:
      print(f"optimized_mod takes {time.time() - enter_ts} seconds overall")

    none_remover.add_nones(result)
    if len(result) == 1:
      return result[0]
    else:
      return result

  return optimized_mod


class FallBackNodeCollector(torch.fx.Interpreter):

  def __init__(self, module):
    super().__init__(module)
    self._fallback_ops = []

  def run_node(self, n: torch.fx.Node):
    metrics.clear_counters()
    result = super().run_node(n)
    fallback_ops = get_fallback_ops()
    if len(fallback_ops) > 0:
      self._fallback_ops.append(n)
    else:
      # if inputs are non-xla tensors, it should be executed on CPU
      if n.op in ["call_function", "call_module", "call_method"]:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        for arg in args:
          if isinstance(arg, torch.Tensor) and not is_xla_tensor(arg):
            self._fallback_ops.append(n)
            break
    return result

  def get_fallback_ops(self):
    return self._fallback_ops


class InputCollector(torch.fx.Interpreter):

  def __init__(self, module):
    super().__init__(module)

  def call_module(self, target, args, kwargs):
    if "fused_" in target:
      submod = self.fetch_attr(target)
      submod.xla_args = args
    return super().call_module(target, args, kwargs)


def extract_compiled_graph(xla_model, xla_args):
  # This call is critical to make sure xla_args' tensor id show up in graph_input_tensor_ids
  xm.mark_step()

  # This logic, needed for supporting in-place operations, is a duplicate of
  # the one in the main `extract_internal` function above. We need to do this
  # check for fetching fallback ops as well.
  # TODO (@wonjoo): Make this duplicate code a bit cleaner.
  xla_args_need_update_bool = torch_xla._XLAC._check_tensor_need_materialization(
      xla_args)

  cloned_xla_args = [
      torch.clone(xla_arg) if isinstance(xla_arg, torch.Tensor) else xla_arg
      for xla_arg in xla_args
  ]

  # execute model once to collect fallback ops
  collector = FallBackNodeCollector(xla_model)
  collector.run(*xla_args)
  fallback_ops = collector.get_fallback_ops()

  class XlaOperatorSupport(torch.fx.passes.operator_support.OperatorSupport):

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
      return node.op in [
          "call_function", "call_module", "call_method"
      ] and (node not in fallback_ops or node.target == operator.getitem)

  # partition the model and exectue to collect inputs
  supported_ops = XlaOperatorSupport()
  partitioner = CapabilityBasedPartitioner(xla_model, supported_ops)
  partitions = partitioner.propose_partitions()
  partitioned_graph = partitioner.fuse_partitions(partitions)
  InputCollector(partitioned_graph).run(*xla_args)

  # Again, same logic in the `extract_internal` above to support in-place operations.
  # TODO (@wonjoo): Make this duplicate code a bit cleaner.
  if xla_args_need_update_bool:
    for xla_arg, cloned_xla_arg in zip(xla_args, cloned_xla_args):
      if isinstance(xla_arg, torch.Tensor):
        xla_arg.copy_(cloned_xla_arg)

  torch_xla._XLAC._clear_pending_irs(str(xm.xla_device()))

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
