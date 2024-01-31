import copy

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo


def _replace_expand_symsize(graph: torch.fx.Graph):
  g = copy.deepcopy(graph)
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
        # new_args[1][dim] = 1 # Use a dummy dim for the dynamic dim
        # new_args[1][dim] = np.iinfo(np.int64).min
        new_args = tuple(new_args)
        dynamic_src = sym_size_node.args[0]
        dynmiac_src_dim = sym_size_node.args[1]
        n.args = new_args + (dynamic_src, dynmiac_src_dim, dim)
        import torch_xla.experimental.xla_dynamic_broadcast
        n.target = torch.ops.xla_dynamic_broadcast.dynamic_expand
        # if len(list(sym_size_node.users)) == 0:
        #   g.erase_node(sym_size_node)
  return g


def replace_dynamic_expand_with_xla_op(ep: "ExportedProgram"):
  ep.graph_module.graph = _replace_expand_symsize(ep.graph_module.graph)

def _replace_symsize_view(graph: torch.fx.Graph):
  ATEN_VIEW_OPS = [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default]
  g = copy.deepcopy(graph)
  for n in g.nodes:
    if n.target in ATEN_VIEW_OPS:
      view_dims = n.args[1]
      sym_sizes = []
      for dim, node in enumerate(view_dims):
        if not isinstance(node, int):
          sym_sizes.append((dim, node))
      print("check sym_sizes: ", sym_sizes)
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
        if hasattr(dynamic_src.target, "__name__") and dynamic_src.target.__name__ == "mul":
          assert isinstance(dynamic_src.args[0], int) or isinstance(dynamic_src.args[1], int)
          mul_node = dynamic_src
          if isinstance(dynamic_src.args[1], int):
            mul_scaler = dynamic_src.args[1]
            sym_size_node = dynamic_src.args[0]
          else:
            mul_scaler = dynamic_src.args[0]
            sym_size_node = dynamic_src.args[1]
        else:
          # import pdb; pdb.set_trace()
          assert dynamic_src.target == torch.ops.aten.sym_size.int
        sym_size_src = sym_size_node.args[0]
        sym_size_dim = sym_size_node.args[1]

        n.args = tuple(new_args) + (sym_size_src, sym_size_dim, 0, int(mul_scaler))
        # Replace with real func
        import torch_xla.experimental.xla_dynamic_broadcast
        n.target = torch.ops.xla_dynamic_broadcast.dynamic_view

        # Remove dangling mul or sym_size node
        if mul_node is not None and len(list(mul_node.users)) == 0:
          g.erase_node(mul_node)
        if len(list(sym_size_node.users)) == 0:
          g.erase_node(sym_size_node)
  return g

def replace_dynamic_view_with_xla_op(ep: "ExportedProgram"):
  ep.graph_module.graph = _replace_symsize_view(ep.graph_module.graph)
