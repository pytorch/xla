import os
import re
import sys
import unittest

import numpy as np
import torch
import torch.utils._pytree as pytree
import torch_xla
import torch_xla.core.xla_model as xm
from torch.export import Dim, export
from torch_xla.experimental.unbounded_dynamism_export import *
from torch_xla.stablehlo import exported_program_to_stablehlo
from utils import wrap_func_as_nn_module


class ExportFxPassTest(unittest.TestCase):

  def test_decompose_dynamic_shape_select(self):
    args = (torch.rand((10, 197, 768)), 1, 0)
    dynamic_shapes = ([{0: Dim("bs")}, None, None],)
    m = wrap_func_as_nn_module(torch.ops.aten.select.int)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep(*args)
    ep.graph_module.graph = decompose_dynamic_shape_select(ep.graph_module)
    self.assertTrue('aten.view' in ep.graph_module.code)
    replace_dynamic_view_with_xla_op(ep.graph_module)
    ep.graph_module.recompile()
    self.assertTrue('aten.view' not in ep.graph_module.code)
    self.assertTrue('xla.dynamic_view' in ep.graph_module.code)
    out2 = ep(*args)
    self.assertTrue(torch.allclose(out1, out2))

  def test_no_op_slice_removal(self):

    class M(torch.nn.Module):

      def forward(self, x):
        x = x * 2
        return torch.ops.aten.slice(x, 1, 0, 9223372036854775807)

    m = M()
    args = (torch.rand((10, 197, 768)),)
    dynamic_shapes = ({0: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep(*args)
    self.assertTrue('aten.slice' in ep.graph_module.code)
    ep.graph_module.graph = remove_no_op_slice(ep.graph_module)
    ep.graph_module.recompile()
    self.assertTrue('aten.slice' not in ep.graph_module.code)
    out2 = ep(*args)
    self.assertTrue(torch.allclose(out1, out2))

  def test_dynamic_view(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, [16, 16])

      def forward(self, x):
        x = self.conv(x)
        return x.view(x.shape[0], x.shape[1], -1)

    m = M()
    args = (torch.rand((10, 3, 224, 224)),)
    dynamic_shapes = ({0: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep(*args)
    replace_dynamic_view_with_xla_op(ep.graph_module)
    ep.graph_module.recompile()
    self.assertTrue('xla.dynamic_view' in ep.graph_module.code)
    out2 = ep(*args)
    self.assertTrue(torch.allclose(out1, out2))

  def test_dynamic_view_multiplier(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, [16, 16])

      def forward(self, x):
        x = self.conv(x)
        return x.view(x.shape[0]*x.shape[1], -1)

    m = M()
    args = (torch.rand((10, 3, 224, 224)),)
    dynamic_shapes = ({0: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep(*args)
    replace_dynamic_view_with_xla_op(ep.graph_module)
    print(ep)
    ep.graph_module.recompile()
    print(ep.graph_module.code)
    self.assertTrue('xla.dynamic_view' in ep.graph_module.code)
    out2 = ep(*args)
    self.assertTrue(torch.allclose(out1, out2))

  def test_dynamic_expand(self):

    class M(torch.nn.Module):

      def forward(self, x, image):
        return x.expand([image.shape[0], -1, -1])

    m = M()
    args = (torch.rand((1, 1, 5)), torch.rand((3, 4)))
    dynamic_shapes = (
        None,
        {
            0: Dim("bs")
        },
    )
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep(*args)
    replace_dynamic_expand_with_xla_op(ep.graph_module)
    ep.graph_module.recompile()
    self.assertTrue('xla.dynamic_expand' in ep.graph_module.code)
    out2 = ep(*args)
    self.assertTrue(torch.allclose(out1, out2))


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
