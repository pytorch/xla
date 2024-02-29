import torch
from torch.fx import Graph

aten = torch.ops.aten


def get_symbolic_dims(shape):
  symbolic_dims = []
  for idx, size in enumerate(shape):
    if not isinstance(size, int):
      symbolic_dims.append(idx)
  return symbolic_dims


def decompose_dynamic_shape_select(graph: Graph):
  for n in graph.nodes:
    if n.op == "call_function" and n.target == aten.select.int:
      import pdb
      # pdb.set_trace()
      select_src_shape = n.args[0].meta['val'].shape
      symbolic_dims = get_symbolic_dims(select_src_shape)
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
          view_node = graph.call_function(aten.view, view_args)
          n.replace_all_uses_with(view_node)
        graph.erase_node(n)
  return graph


def _is_no_op_slice(n):
  assert n.op == "call_function" and n.target == aten.slice.Tensor
  return n.args[2] == 0 and n.args[3] == torch.iinfo(torch.int64).max


def remove_no_op_slice(graph: Graph):
  for n in graph.nodes:
    if n.op == "call_function" and n.target == aten.slice.Tensor:
      if _is_no_op_slice(n):
        slice_src_node = n.args[0]
        n.replace_all_uses_with(slice_src_node)
      graph.erase_node(n)
  return graph
