import unittest
import re

import torch
import torch_xla
import torch_xla.experimental.xla_mlir_debuginfo
from torch_xla.stablehlo import exported_program_to_stablehlo


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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
