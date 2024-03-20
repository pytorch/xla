import torch
import torch_xla.experimental.xla_dynamic_reshape_ops
from torch.export import export
from torch.fx import Graph, GraphModule, subgraph_rewriter

aten = torch.ops.aten


def wrap_func_as_nn_module(f):

  class M(torch.nn.Module):

    def forward(self, *args):
      return f(*args)

  return M()


def native_layer_norm_impl(input, normalized_shape, weight, bias, eps):
  mean = torch.mean(input, -1, keepdim=True)
  rstd = torch.rsqrt(torch.var(input, -1, keepdim=True, correction=0) + eps)
  out = ((input + mean * -1) * rstd)  # sub doesn't support unbounded dynamism
  out = out * weight + bias
  return out


def native_group_norm_impl(input, weight, bias, N, C, HxW, group, eps):
  mean = torch.mean(input, -1, keepdim=True)
  rstd = torch.rsqrt(torch.var(input, -1, keepdim=True, correction=0) + eps)
  out = ((input + mean * -1) * rstd)  # sub doesn't support unbounded dynamism
  weight = weight.unsqueeze(1)
  bias = bias.unsqueeze(1)
  out = out * weight + bias
  return out


def native_layer_norm_pattern(x, dim, weight, bias, eps):
  # We need to include get_item semantic in the pattern.
  # native_layer_norm returns a tuple, which is a single fx node if traced
  # in export, but it will be multiple fx nodes if graph is symbolic traced.
  #
  # If we export the native_layer_norm pattern it doesn't work with
  # subgraph_rewritter, because of the dangling input for the eps constant.
  # Exporting a nn.Module with a nn.LayerNorm inside also doesn't work,
  # the order of the LayerNorm parameters in the exported grpah signature
  # may not match the order in the replacement function.
  return torch.ops.aten.native_layer_norm.default(x, dim, weight, bias, eps)[0]


def native_group_norm_pattern(x, weight, bias, N, C, HxW, group, eps):
  return torch.ops.aten.native_group_norm.default(x, weight, bias, N, C, HxW,
                                                  group, eps)[0]


def decompose_dynamic_native_layer_norm(gm):
  replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(
      gm,
      native_layer_norm_pattern,
      native_layer_norm_impl,
      ignore_literals=False)
  # Only support normalize along the last dim now. Check if replacement is valid.
  for matches in replaced_patterns:
    for pattern_n, match_n in matches.nodes_map.items():
      if isinstance(match_n, list):
        assert len(match_n) == 1


def decompose_dynamic_native_group_norm(gm):
  replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(
      gm,
      native_group_norm_pattern,
      native_group_norm_impl,
      ignore_literals=True)


def decompose_dynamic_shape_select(gm: GraphModule):
  '''
  Decompose `aten.select.int` with symbolic input shape.

  1. `aten.slice` over the target dim on the target index.
  2. `aten.view` to squeeze the sliced dim.

  The symbolic shape in `aten.view` will be handled by `aten.view` -> `xla.dynamic_view` pass.
  '''
  graph = gm.graph
  for n in graph.nodes:
    if n.op == "call_function" and n.target == aten.select.int:
      select_src_shape = n.args[0].meta['val'].shape
      symbolic_dims = [
          i for i, x in enumerate(select_src_shape) if not isinstance(x, int)
      ]
      if len(symbolic_dims) > 0:
        assert len(symbolic_dims) == 1, "Only 1 dimention can be symbolic."
        with graph.inserting_before(n):
          select_src_node = n.args[0]
          select_dim = n.args[1]
          assert symbolic_dims[
              0] != select_dim, "Selected dim cannot be symbolic."
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
  '''
  Remove no-op slice in FX Graph.
  '''
  graph = gm.graph
  for n in graph.nodes:
    if n.op == "call_function" and n.target == aten.slice.Tensor:
      if _is_no_op_slice(n):
        slice_src_node = n.args[0]
        n.replace_all_uses_with(slice_src_node)
        graph.erase_node(n)


def replace_dynamic_expand_with_xla_op(gm: GraphModule):
  '''
  Replace `aten.expand` with symbolic input shape with `xla.dynamic_expand`.
  `xla.dynamic_expand` takes additional args for the source of the symbolic
  dimension.
  '''
  g = gm.graph
  for n in g.nodes:
    if n.target == torch.ops.aten.expand.default:
      expand_dims = n.args[1]
      symbolic_dims_sizes = []
      for dim, node in enumerate(expand_dims):
        if not isinstance(node, int):
          symbolic_dims_sizes.append((dim, node))
      if len(symbolic_dims_sizes) == 0:
        continue
      assert len(symbolic_dims_sizes) == 1
      src_sizes = n.args[0].meta['val'].size()
      expanded_sizes = n.args[1]
      assert len(src_sizes) == len(expanded_sizes)
      for i in range(len(src_sizes)):
        if not isinstance(src_sizes[i], int) and not isinstance(
            expanded_sizes[i], int):
          assert src_sizes[i] == expanded_sizes[i].meta[
              'val'], "Expanded symbolic dim to a different symbolic size is not supported."
      for dim, sym_size_node in symbolic_dims_sizes:
        assert sym_size_node.op == "call_function" and sym_size_node.target == aten.sym_size.int
        dynamic_src = sym_size_node.args[0]
        dynmiac_src_dim = sym_size_node.args[1]
        n.args = n.args + (dynamic_src, dynmiac_src_dim, dim)
        n.target = torch.ops.xla.dynamic_expand


def replace_dynamic_view_with_xla_op(gm: GraphModule):
  '''
  Replace `aten.view` with symbolic input shape with `xla.dynamic_view`.
  `xla.dynamic_view` takes additional args for the source of the symbolic
  dimension.
  '''
  g = gm.graph
  ATEN_VIEW_OPS = [
      torch.ops.aten.view,
      torch.ops.aten.view.default,
      torch.ops.aten._unsafe_view,
      torch.ops.aten._unsafe_view.default,
  ]
  for n in g.nodes:
    if n.target in ATEN_VIEW_OPS:
      view_dims = n.args[1]
      sym_sizes = []
      for dim, node in enumerate(view_dims):
        if not isinstance(node, int):
          sym_sizes.append((dim, node))
      if len(sym_sizes) != 0:
        assert len(sym_sizes
                  ) == 1, "Only 1 symoblic dim in view src tensor is supported."
        new_args = list(n.args)
        new_args[1] = list(new_args[1])
        target_dim = sym_sizes[0][0]
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
          if isinstance(mul_node.args[1], int):
            mul_scaler = mul_node.args[1]
            sym_size_node = mul_node.args[0]
          else:
            mul_scaler = mul_node.args[0]
            sym_size_node = mul_node.args[1]
        else:
          assert dynamic_src.target == torch.ops.aten.sym_size.int
        sym_size_src = sym_size_node.args[0]
        sym_size_dim = sym_size_node.args[1]
        n.args = tuple(new_args) + (sym_size_src, sym_size_dim, target_dim,
                                    int(mul_scaler))
        n.target = torch.ops.xla.dynamic_view


def dynamic_unsqueeze_to_view(gm: GraphModule):
  '''
  Replace unsqueeze with symbolic input shape to view.
  '''
  graph = gm.graph
  for n in graph.nodes:
    if n.op == "call_function" and n.target == torch.ops.aten.unsqueeze.default:
      unsqueeze_src = n.args[0]
      src_shape = unsqueeze_src.meta['val'].shape
      symbolic_dims = [
          i for i, x in enumerate(src_shape) if not isinstance(x, int)
      ]
      if len(symbolic_dims) == 0:
        continue
      view_args = list(src_shape)
      with graph.inserting_before(n):
        for dim in symbolic_dims:
          get_size_node = graph.call_function(aten.sym_size.int,
                                              (unsqueeze_src, dim))
          view_args[dim] = get_size_node
        squeezed_dim = n.args[1]
        view_args = (unsqueeze_src,
                     view_args[:squeezed_dim] + [1] + view_args[squeezed_dim:])
        view_node = graph.call_function(aten.view, view_args)
        n.replace_all_uses_with(view_node)
        graph.erase_node(n)


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
      decompose_dynamic_native_group_norm,
      decompose_dynamic_native_layer_norm,
      dynamic_unsqueeze_to_view,
      replace_dynamic_expand_with_xla_op,
      replace_dynamic_view_with_xla_op,
  ]
  for p in passes:
    p(ep.graph_module)
    ep.graph_module.graph.eliminate_dead_code()
    ep.graph_module.graph.lint()
    ep.graph_module.recompile()
