import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch_xla
import torch_xla.experimental.xla_marker
from torch.export import ExportedProgram
from torch.export.graph_signature import TensorArgument
from torch.fx import Graph, GraphModule, Node, subgraph_rewriter
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.utils import _pytree as pytree
from torch_xla.core import xla_model as xm

__all__ = ["mark_pattern"]


@dataclass
class PortTag:
  name: str  # Identify Patttern
  pos: int  # Arg/return position
  id: int  # Patten instance id
  is_input: bool = True  # If the tagged tensor is input/output
  attr: Dict = None  # Attribute of the pattern, only output has attr field


class TagSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj)
    return super().default(obj)


@dataclass
class ConstAttrTracker:
  attr_name: str
  pattern_arg_pos: int = -1
  pattern_arg_key: str = ""
  transform: Callable = lambda x: x
  inverse_transform: Callable = lambda x: x
  source_targets: List[Tuple[Any,
                             Any]] = dataclasses.field(default_factory=list)

  def _is_equal(self, x: Any, y: Any):
    if type(x) != type(y):
      return False
    if type(x) in [int, str]:
      return x == y
    if isinstance(x, float):
      rel_tol = 1e-07
      abs_tol = 0.0
      return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
    if isinstance(x, list):
      if len(x) != len(y):
        return False
      return all([self._is_equal(a, b) for a, b in zip(x, y)])

    raise Exception(f"Cannot compare type: {type(x)}")

  def track(self, *sources):
    for source in sources:
      target = self.transform(source)
      if not self._is_equal(self.inverse_transform(target), source):
        raise Exception(
            f"Invalid transform/inverse_transform for {self.attr_name}")
      self.source_targets.append([source, target])
    return self


@dataclass
class ConstAttrLoc:
  tracker: ConstAttrTracker
  node_name: str
  pos: Union[int, str]


def extract_and_replace_const_from_matched_pattern(matches: List[InternalMatch],
                                                   loc: ConstAttrLoc):
  val = None
  matched_const_vals = []
  for match in matches:
    for k, v in match.nodes_map.items():
      if k.name == loc.node_name:
        if loc.pos is not None:
          if isinstance(loc.pos, int):
            val = v.args[loc.pos]
          else:
            if loc.pos in v.kwargs.keys():
              val = v.kwargs[loc.pos]
    pattern_arg_val = loc.tracker.inverse_transform(val)
    matched_const_vals.append(pattern_arg_val)
  return loc.tracker.attr_name, matched_const_vals


def find_const_attr_loc(pattern, pattern_args, tracker: ConstAttrTracker):
  const_loc_intersections = None
  for source, target in tracker.source_targets:
    track_args = list(pattern_args)
    track_args[tracker.pattern_arg_pos] = source
    ep = torch.export.export(pattern, tuple(track_args))
    ep.graph_module.graph.print_tabular()

    const_locs = set()
    nodes = ep.graph_module.graph.nodes
    for n in nodes:
      for arg_pos, arg in enumerate(n.args):
        if type(arg) == type(target) and arg == target:
          const_locs.add((n.name, arg_pos))
      for attr, val in n.kwargs.items():
        if type(val) == type(target) and val == target:
          const_locs.add((n.name, attr))

    if const_loc_intersections is None:
      const_loc_intersections = const_locs
    else:
      const_loc_intersections = const_loc_intersections & const_locs

    if not const_loc_intersections:
      break

  if not const_loc_intersections:
    return None
  # Choose any occurrence as the attr provider
  node_name, pos = const_loc_intersections.pop()
  return ConstAttrLoc(tracker, node_name, pos)


def eliminate_clone_ops(graph: Graph):
  # torch export adds additional aten.clone nodes to produce contiguous in memory tensors
  # depending on tensor sizes for runtime efficiency. However, these unpredictable clone
  # nodes can break the pattern matching. Thus remove all clones in model and pattern graphs.
  for n in graph.nodes:
    if n.op == "call_function" and n.name.startswith("clone"):
      n.replace_all_uses_with(n.args[0])
      graph.erase_node(n)


def eliminate_dangling_arg(graph: Graph):
  nodes_to_erase = []
  for n in graph.nodes:
    if n.op == "placeholder" and len(n.users) == 0:
      nodes_to_erase.append(n)
  for n in nodes_to_erase:
    graph.erase_node(n)


def canonicalize_exported_graph(graph: Graph):
  eliminate_dangling_arg(graph)
  eliminate_clone_ops(graph)


def get_node_from_name(graph: Graph, name: str) -> Node:
  for n in graph.nodes:
    if n.name == name:
      return n
  return None


def get_pattern_placeholder_in_order(pattern_ep: ExportedProgram):
  # Exclude const arg
  return [
      spec.arg.name
      for spec in pattern_ep.graph_signature.input_specs
      if isinstance(spec.arg, TensorArgument)
  ]


def get_pattern_outputs_in_order(pattern_ep: ExportedProgram):
  return [spec.arg.name for spec in pattern_ep.graph_signature.output_specs]


def insert_marker(graph: Graph, node: Node, tag: PortTag):
  with graph.inserting_after(node):
    n = graph.call_function(
        torch.ops.xla_pattern_marking.mark_tensor,
        args=(node, tag.name, tag.pos, tag.id, tag.is_input, tag.attr),
    )
    node.replace_all_uses_with(n)
    n.update_arg(0, node)
    return n


def mark_pattern(
    pattern_name: str,
    exported_ep: GraphModule,
    pattern: Union[Callable, torch.nn.Module],
    pattern_args: Tuple = None,
    pattern_attrs: Optional[Dict[str, Any]] = dict(),
    const_attr_trackers: Optional[List[Union[ConstAttrLoc,
                                             ConstAttrTracker]]] = None,
):
  exported_ep = copy.deepcopy(exported_ep)
  orig_graph = exported_ep.graph_module.graph

  pattern_ep = torch.export.export(pattern, pattern_args)

  pattern_graph = pattern_ep.graph_module.graph
  # canonicalize_exported_graph(orig_graph)
  canonicalize_exported_graph(pattern_graph)

  matcher = SubgraphMatcher(
      pattern_ep.graph_module.graph,
      match_output=False,
      match_placeholder=False,
      remove_overlapping_matches=True,
      ignore_literals=True,
  )

  matches: List[InternalMatch] = matcher.match(exported_ep.graph_module.graph)

  const_attr_dicts = dict()  # name -> list[matched_vals]
  if const_attr_trackers:
    for tracker in const_attr_trackers:
      if isinstance(tracker, ConstAttrLoc):
        loc = tracker
      else:
        loc = find_const_attr_loc(pattern, pattern_args, tracker)
      assert loc is not None
      print(loc)
      attr_name, vals = extract_and_replace_const_from_matched_pattern(
          matches, loc)
      const_attr_dicts[attr_name] = vals

  placeholder_list = get_pattern_placeholder_in_order(pattern_ep)
  output_list = get_pattern_outputs_in_order(pattern_ep)
  for match_idx, m in enumerate(matches):
    # Get tracked const vals
    attr = copy.deepcopy(pattern_attrs)
    for attr_name, vals in const_attr_dicts.items():
      attr[attr_name] = vals[match_idx]
    for idx, placeholder in enumerate(placeholder_list):
      matched_placeholder = m.nodes_map[get_node_from_name(
          pattern_graph, placeholder)]
      tag = PortTag(pattern_name, idx, match_idx, is_input=True, attr=attr)
      insert_marker(orig_graph, matched_placeholder, tag)

    for idx, output in enumerate(output_list):
      matched_output = m.nodes_map[get_node_from_name(pattern_graph, output)]
      tag = PortTag(pattern_name, idx, match_idx, is_input=False, attr=attr)
      n = insert_marker(orig_graph, matched_output, tag)
      # If marking node is inserted after output, update graph_signature
      for output_spec in exported_ep.graph_signature.output_specs:
        if isinstance(
            output_spec.arg,
            TensorArgument) and output_spec.arg.name == matched_output.name:
          output_spec.arg.name = n.name
  orig_graph.lint()
  orig_graph.print_tabular()
  return exported_ep


def xla_add_tag(tensor: torch.Tensor, tag: str) -> torch.Tensor:
  if isinstance(tensor,
                torch.Tensor) and tensor.get_device() == xm.xla_device():
    torch_xla._XLAC._xla_add_tag(tensor, tag)
  return tensor


def fx_node_add_tag(n, tag):
  return pytree.tree_map_only(torch.Tensor, lambda t: xla_add_tag(t, tag), n)


def mark_model_explorer_loc(exported_ep: GraphModule):

  def class_fullname(klass):
    module = klass.__module__
    if module == "builtins":
      return klass.__qualname__
    return module + "." + klass.__qualname__

  graph = exported_ep.graph_module.graph

  for n in list(graph.nodes):
    if n.op != "call_function":
      continue

    if "nn_module_stack" not in n.meta:
      return n

    nn_module_stack = n.meta["nn_module_stack"]

    layers = []
    for var, (path, cls) in nn_module_stack.items():
      iid = path.split(".")[-1]
      layer = class_fullname(cls)
      if iid != "":
        layer = f"{layer}__{iid}"
      layers.append(layer)
    layers.append("aten__" + n.name)

    layers_loc = "/".join(layers)

    users = list(n.users)
    with graph.inserting_after(n):
      add_tag_node = graph.call_function(
          fx_node_add_tag, args=(n, json.dumps({"layers_loc": layers_loc})))
      for user in users:
        user.replace_input_with(n, add_tag_node)
  return exported_ep
