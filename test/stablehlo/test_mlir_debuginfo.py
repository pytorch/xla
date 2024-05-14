import re
import unittest

import torch
import torch_xla
import torch_xla.experimental.xla_mlir_debuginfo
from torch_xla.stablehlo import (StableHLOExportOptions,
                                 exported_program_to_stablehlo)


class XlaMlirDebuginfoTest(unittest.TestCase):

  def test_write_debuginfo(self):

    class SampleModel(torch.nn.Module):

      def forward(self, x, y):
        x = x + y
        x = torch.ops.xla.write_mlir_debuginfo(x, "MY_ADD")
        x = x - y
        x = torch.ops.xla.write_mlir_debuginfo(x, "MY_SUB")
        return x

    model = SampleModel()
    exported_program = torch.export.export(model,
                                           (torch.rand(10), torch.rand(10)))
    mlir_text = exported_program_to_stablehlo(
        exported_program).get_stablehlo_text()
    self.assertTrue(re.search(r'stablehlo.add.+\"MY_ADD\"', mlir_text))
    self.assertTrue(re.search(r'stablehlo.sub.+\"MY_SUB\"', mlir_text))

  def test_export_node_metadata(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=4, out_features=16, bias=True)
        self.fc2 = torch.nn.Linear(in_features=16, out_features=10, bias=True)

      def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.relu(x)

    args = (torch.rand(2, 4),)
    ep = torch.export.export(M(), args)
    export_options = StableHLOExportOptions()
    export_options.export_node_metadata = True
    shlo = exported_program_to_stablehlo(ep, options=export_options)
    shlo_text = shlo.get_stablehlo_text()
    print(shlo_text)
    self.assertTrue('stack_trace' in shlo_text)
    self.assertTrue('nn_module_stack' in shlo_text)
    self.assertTrue('source_fn_stack' in shlo_text)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
