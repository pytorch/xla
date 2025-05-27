import re
import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo

device = torch.device('xla')


class XLAExportInterpreterTest(unittest.TestCase):

  def test_constant_wrapping(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return 1.0 - x

    ep = torch.export.export(M(), (torch.rand(2, 3),))
    ep = ep.run_decompositions()
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'stablehlo.constant.* : tensor<2x3xf32>', shlo_text)
        is not None)

  def test_constant_wrapping_scalar_variant(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.rsub(x, 1.0)

    ep = torch.export.export(M(), (torch.rand(2, 3),))
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'stablehlo.constant.* : tensor<2x3xf32>', shlo_text)
        is not None)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
