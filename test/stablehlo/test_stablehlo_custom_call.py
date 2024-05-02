import re
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.experimental.stablehlo_custom_call
from torch.library import Library, impl, impl_abstract
from torch_xla.experimental.stablehlo_custom_call import stablehlo_custom_call
from torch_xla.stablehlo import (StableHLOExportOptions,
                                 exported_program_to_stablehlo)

m = Library("my_custom_library", "DEF")


class StableHLOCustomCallExportTest(unittest.TestCase):

  def test_single_output(self):

    m.define("custom_op(Tensor input) -> Tensor")

    @impl(m, "custom_op", "Meta")
    def custom_op_meta(x):
      return torch.empty_like(x)

    class M(torch.nn.Module):

      def forward(self, x):
        x = torch.sin(x)
        x = torch.ops.my_custom_library.custom_op(x)
        x = torch.cos(x)
        x = torch.ops.my_custom_library.custom_op(x)
        x = torch.sin(x)
        return x

    options = StableHLOExportOptions()
    options.custom_ops_allowed_in_graph.add("my_custom_library")
    ep = torch.export.export(M(), (torch.randn(3, 3),))
    shlo_module = exported_program_to_stablehlo(ep, options)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"stablehlo.custom_call.*@my_custom_library\.custom_op\.default",
            shlo_text) is not None)
    self.assertTrue(
        re.search(r"tensor<3x3xf32>.*->.*tensor<3x3xf32>", shlo_text)
        is not None)
    self.assertTrue(shlo_text.count("@my_custom_library.custom_op.default", 2))

  def test_multiple_input_output(self):

    m.define("custom_op2(Tensor input, Tensor input) -> (Tensor, Tensor)")

    @impl(m, "custom_op2", "Meta")
    def custom_op2_meta(x, y):
      return torch.empty_like(x), torch.empty(y.shape[1:], device='meta')

    class M(torch.nn.Module):

      def forward(self, x, y):
        x = torch.sin(x)
        x, y = torch.ops.my_custom_library.custom_op2(x, y)
        x = torch.cos(x)
        x, y = torch.ops.my_custom_library.custom_op2(x, y)
        y = torch.sin(y)
        return x, y

    options = StableHLOExportOptions()
    options.custom_ops_allowed_in_graph.add("my_custom_library")
    ep = torch.export.export(M(), (torch.randn(3, 3), torch.randn(5, 5)))
    shlo_module = exported_program_to_stablehlo(ep, options)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"stablehlo.custom_call.*@my_custom_library\.custom_op2\.default",
            shlo_text) is not None)
    self.assertTrue(
        re.search(
            r"tensor<3x3xf32>.*tensor<5x5xf32>.*->.*tuple<tensor<3x3xf32>, tensor<5xf32>>",
            shlo_text) is not None)
    self.assertTrue(shlo_text.count("@my_custom_library.custom_op2.default", 2))

  def test_stable_custom_call_api(self):

    m.define("custom_op3(Tensor input) -> Tensor")

    @impl(m, "custom_op3", "Meta")
    def custom_op3_meta(x):
      return torch.empty(x.shape[1:], device='meta')

    @impl(m, "custom_op3", "XLA")
    def custom_op3_xla(x):
      res = stablehlo_custom_call((x,), "custom_op3", [x.shape[1:]],
                                  [torch.int8], True, "backend_config", 1)
      return res

    class M(torch.nn.Module):

      def forward(self, x):
        x = torch.sin(x)
        x = torch.ops.my_custom_library.custom_op3(x)
        x = torch.cos(x)
        return x

    ep = torch.export.export(M(), (torch.randn(3, 3),))
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"stablehlo.custom_call.*@custom_op3", shlo_text) is not None)
    self.assertTrue(
        re.search(r"tensor<3x3xf32>.*->.*tensor<3xi8>", shlo_text) is not None)
    self.assertTrue("backend_config = \"backend_config\"" in shlo_text)
    self.assertTrue("has_side_effect = true" in shlo_text)
    # TODO: api version lost during conversion, or not shown in txt format.
    # self.assertTrue("api_version = 1" in shlo_text)


if __name__ == "__main__":

  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
