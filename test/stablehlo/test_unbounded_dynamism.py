import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo

# Note: Unbounded dynamism is under development. It works with unmerged
# XLA changes. Experimental XLA branch: https://github.com/lsy323/openxla-xla/tree/lsiyuan/sandeep-dynamism-rebased

device = xm.xla_device()


class UnboundedDynamismExportTest(unittest.TestCase):

  def test_simply_add(self):
    a = torch.tensor([[1, 2], [2, 4]], device=device)
    torch_xla._XLAC._xla_mark_dynamic(a, 0)
    b = torch.tensor([[1, 2], [2, 4]], device=device)
    torch_xla._XLAC._xla_mark_dynamic(b, 0)
    c = a * b
    hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
    self.assertTrue(
        "(p0.1: s64[?,2], p1.2: s64[?,2]) -> (s64[?,2])" in hlo_content)

  def test_export_dynamism(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, x, y):
        return x * y

    example_args = (torch.tensor([[1, 2], [2, 4]], device=device),
                    torch.tensor([[1, 2], [2, 4]], device=device))
    constraints = [
        # First dimension of each input is a dynamic batch size
        torch.export.dynamic_dim(example_args[0], 0),
        torch.export.dynamic_dim(example_args[1], 0),
        # The dynamic batch size between the inputs are equal
        torch.export.dynamic_dim(example_args[0],
                                 0) == torch.export.dynamic_dim(
                                     example_args[1], 0),
    ]
    ep = torch.export.export(M(), args=example_args, constraints=constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text("forward")
    self.assertTrue(
        "(%arg0: tensor<?x2xi64>, %arg1: tensor<?x2xi64>) -> tensor<?x2xi64>" in
        shlo_text)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
