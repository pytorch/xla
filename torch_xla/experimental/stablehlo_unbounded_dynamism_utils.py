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
        new_args[1][dim] = 1
        # new_args[1][dim] = np.iinfo(np.int64).min
        new_args = tuple(new_args)
        dynamic_src = sym_size_node.args[0]
        dynmiac_src_dim = sym_size_node.args[1]
        n.args = new_args + (dynamic_src, dynmiac_src_dim, dim)
        import torch_xla.experimental.xla_dynamic_broadcast
        n.target = torch.ops.xla_dynamic_broadcast.dynamic_expand
        if len(list(sym_size_node.users)) == 0:
          g.erase_node(sym_size_node)
  return g


def replace_dynamic_expand_with_xla_op(ep: "ExportedProgram"):
  ep.graph_module.graph = _replace_expand_symsize(ep.graph_module.graph)
