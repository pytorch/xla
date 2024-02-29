import re
import sys
import unittest

import torch
import torch_xla
from torch.export import Dim, export
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo
from utils import wrap_func_as_nn_module
from torch_xla.experimental.unbounded_dynamism_export import *

device = xm.xla_device()


class ExportFxPassTest(unittest.TestCase):

  def test_decompose_dynamic_shape_select(self):
    args = (torch.rand((10, 197, 768)), 1, 0)
    dynamic_shapes = ([{0: Dim("bs")}, None, None],)
    m = wrap_func_as_nn_module(torch.ops.aten.select.int)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    # ep.graph_module.graph.print_tabular()
    out1 = ep(*args)
    ep.graph_module.graph = decompose_dynamic_shape_select(
        ep.graph_module.graph)
    # ep.graph_module.graph.print_tabular()
    out2 = ep(*args)
    torch.allclose(out1, out2)

  def test_no_op_slice_removal(self):

    class M(torch.nn.Module):

      def forward(self, x):
        x = x * 2
        return torch.ops.aten.slice(x, 1, 0, 9223372036854775807)

    m = M()
    args = (torch.rand((10, 197, 768)),)
    dynamic_shapes = ({0: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    # ep.graph_module.graph.print_tabular()
    out1 = ep(*args)
    ep.graph_module.graph = remove_no_op_slice(ep.graph_module.graph)
    # ep.graph_module.graph.print_tabular()
    out2 = ep(*args)
    torch.allclose(out1, out2)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
