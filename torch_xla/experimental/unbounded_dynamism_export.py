import torch
import torch_xla.experimental.xla_dynamic_reshape_ops
from torch.fx import Graph, GraphModule, subgraph_rewriter

aten = torch.ops.aten

def wrap_func_as_nn_module(f):
  class M(torch.nn.Module):
    def forward(self, *args):
      return f(*args)
  return M()

def native_layer_norm_impl(input, normalized_shape, weight, bias, eps):
  input_shape = input.shape
  # assert len(normalized_shape) == 1 and normalized_shape[0] == input_shape[-1]
  mean = torch.mean(input, -1, keepdim=True)
  rstd = torch.rsqrt(torch.var(input, -1, keepdim=True, correction=0) + eps)
  out = ((input + mean * -1) * rstd) # sub doesn't support unbounded dynamism
  out = out * weight + bias
  return out

def native_layer_norm_pattern(x, dim, weight, bias, eps):
  # Subgraph rewritter doesn't work for pattern with multiple returns.
  return torch.ops.aten.native_layer_norm.default(x, dim, weight, bias, eps)[0]

def decompose_dynamic_native_layer_norm(gm):
  replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(gm, native_layer_norm_pattern, native_layer_norm_impl, ignore_literals=False)
  # Only support normalize along the last dim now.
  for matches in replaced_patterns:
    for pattern_n, match_n in matches.nodes_map.items():
      if isinstance(match_n, list):
        assert len(match_n) == 1

def decompose_dynamic_shape_select(gm: GraphModule):
  graph = gm.graph
  for n in graph.nodes:
    if n.op == "call_function" and n.target == aten.select.int:
      select_src_shape = n.args[0].meta['val'].shape
      symbolic_dims = [
          i for i, x in enumerate(select_src_shape) if not isinstance(x, int)
      ]
      if len(symbolic_dims) > 0:
        with graph.inserting_before(n):
          select_src_node = n.args[0]
          select_dim = n.args[1]
          select_idx = n.args[2]
          slice_args = (select_src_node, select_dim, select_idx,
                        (select_idx + 1), 1)
          slice_node = graph.call_function(aten.slice, slice_args)
          view_new_shape = []
          for dim, size in enumerate(select_src_shape):
            if dim == select_dim:
              continue
            if isinstance(size, int):
              view_new_shape.append(size)
            else:
              get_dim_size_args = (select_src_node, dim)
              get_dim_size_node = graph.call_function(aten.sym_size.int,
                                                      get_dim_size_args)
              view_new_shape.append(get_dim_size_node)
          view_args = (slice_node, view_new_shape)
          view_node = graph.call_function(aten.view.default, view_args)
          n.replace_all_uses_with(view_node)
        graph.erase_node(n)


def _is_no_op_slice(n):
  assert n.op == "call_function" and n.target == aten.slice.Tensor
  return n.args[2] == 0 and n.args[3] == torch.iinfo(torch.int64).max


def remove_no_op_slice(gm: GraphModule):
  graph = gm.graph
  for n in graph.nodes:
    if n.op == "call_function" and n.target == aten.slice.Tensor:
      if _is_no_op_slice(n):
        slice_src_node = n.args[0]
        n.replace_all_uses_with(slice_src_node)
      graph.erase_node(n)
  return graph


def replace_dynamic_expand_with_xla_op(gm: GraphModule):
  g = gm.graph
  for n in g.nodes:
    if n.target == torch.ops.aten.expand.default:
      expand_dims = n.args[1]
      sym_sizes = []
      for dim, node in enumerate(expand_dims):
        if not isinstance(node, int):
          sym_sizes.append((dim, node))
      if len(sym_sizes) == 0:
        return g
      assert len(sym_sizes) == 1

      new_args = list(n.args)
      for dim, sym_size_node in sym_sizes:
        new_args[1] = list(new_args[1])
        new_args = tuple(new_args)
        dynamic_src = sym_size_node.args[0]
        dynmiac_src_dim = sym_size_node.args[1]
        n.args = new_args + (dynamic_src, dynmiac_src_dim, dim)
        n.target = torch.ops.xla.dynamic_expand


def replace_dynamic_view_with_xla_op(gm: GraphModule):
  g = gm.graph
  ATEN_VIEW_OPS = [
      torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default
  ]
  for n in g.nodes:
    if n.target in ATEN_VIEW_OPS:
      view_dims = n.args[1]
      sym_sizes = []
      for dim, node in enumerate(view_dims):
        if not isinstance(node, int):
          sym_sizes.append((dim, node))
      if len(sym_sizes) != 0:
        assert len(sym_sizes) == 1
        new_args = list(n.args)
        # Convert Immutable List to regular list for modification.
        new_args[1] = list(new_args[1])
        # Assume sym size is at batch dim.
        target_dim = 0
        dynamic_src = new_args[1][target_dim]
        # Check mul between sym_size and view.
        mul_scaler = 1
        sym_size_node = dynamic_src
        mul_node = None
        if hasattr(dynamic_src.target,
                   "__name__") and dynamic_src.target.__name__ == "mul":
          assert isinstance(dynamic_src.args[0], int) or isinstance(
              dynamic_src.args[1], int)
          mul_node = dynamic_src
          if isinstance(dynamic_src.args[1], int):
            mul_scaler = dynamic_src.args[1]
            sym_size_node = dynamic_src.args[0]
          else:
            mul_scaler = dynamic_src.args[0]
            sym_size_node = dynamic_src.args[1]
        else:
          assert dynamic_src.target == torch.ops.aten.sym_size.int
        sym_size_src = sym_size_node.args[0]
        sym_size_dim = sym_size_node.args[1]
        n.args = tuple(new_args) + (sym_size_src, sym_size_dim, 0,
                                    int(mul_scaler))
        # Replace with real func
        n.target = torch.ops.xla.dynamic_view
        # Remove dangling mul or sym_size node
        if mul_node is not None and len(list(mul_node.users)) == 0:
          g.erase_node(mul_node)
        if len(list(sym_size_node.users)) == 0:
          g.erase_node(sym_size_node)


def exported_program_has_symbolic_input_shape(ep):
  for n in ep.graph_module.graph.nodes:
    if n.op == "placeholder":
      fake_t = n.meta['val']
      if hasattr(fake_t, 'shape'):
        dynamic_dims = [
            i for i, x in enumerate(fake_t.shape) if not isinstance(x, int)
        ]
        if len(dynamic_dims) > 0:
          return True
  return False


def process_exported_program_with_symbolic_input(ep):
  passes = [
      decompose_dynamic_shape_select,
      remove_no_op_slice,
      decompose_dynamic_native_layer_norm,
      replace_dynamic_expand_with_xla_op,
      replace_dynamic_view_with_xla_op,
  ]
  for p in passes:
    p(ep.graph_module)
    ep.graph_module.graph.eliminate_dead_code()
    ep.graph_module.graph.lint()
    ep.graph_module.recompile()
