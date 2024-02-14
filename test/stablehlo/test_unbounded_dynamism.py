import re
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

  def _test_export_dynamism_wrapper(self, f, args, constraints):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, *args):
        return f(*args)

    m = M()
    ep = torch.export.export(m, args=args, constraints=constraints)
    return ep

  def test_add(self):
    args = (torch.rand((10, 197, 768)), torch.rand((10, 197, 768)))
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
        torch.export.dynamic_dim(args[1], 0),
        torch.export.dynamic_dim(args[0],
                                 0) == torch.export.dynamic_dim(args[1], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.add.Tensor, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<\?x197x768xf32>.*tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)

  def test_add_scalar(self):
    args = (torch.rand((10, 197, 768)), 0.345)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.add.Tensor, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<f32>.*tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)

  @unittest.skip("Unbounded Dynamism not supported on addmm.")
  def test_addmm(self):
    args = (torch.rand((5)), torch.rand((10, 5)), torch.rand((5, 5)))
    constraints = [
        torch.export.dynamic_dim(args[1], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.addmm.default, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x5xf32>.*->.*tensor<\?x5xf32>', shlo_text)
        is not None)

  def test_bmm(self):
    args = (
        torch.rand((24, 197, 64)),
        torch.rand((24, 64, 197)),
    )
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
        torch.export.dynamic_dim(args[1], 0),
        torch.export.dynamic_dim(args[0],
                                 0) == torch.export.dynamic_dim(args[1], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.bmm.default, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<\?x64x197xf32>.*%arg.: tensor<\?x197x64xf32>.*->.*tensor<\?x197x197xf32>',
            shlo_text) is not None)

  def test_cat(self):
    args = ([torch.rand((10, 1, 768)), torch.rand((10, 196, 768))], 1)
    constraints = [
        torch.export.dynamic_dim(args[0][0], 0),
        torch.export.dynamic_dim(args[0][1], 0),
        torch.export.dynamic_dim(args[0][0],
                                 0) == torch.export.dynamic_dim(args[0][1], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.cat.default, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<\?x196x768xf32>.*%arg.: tensor<\?x1x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)

  @unittest.skip("Unbounded Dynamism not supported on conv.")
  def test_conv(self):
    args = (
        torch.rand((10, 3, 224, 224)),
        torch.rand((5, 3, 16, 16)),
        torch.rand((5)),
        [16, 16],
        [0, 0],
        [1, 1],
        False,
        [0, 0],
        1,
    )
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
        torch.export.dynamic_dim(args[0], 0) < 16,
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.convolution.default,
                                            args, constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x3x224x224xf32>.*->.*tensor<\?x5x14x14xf32>',
                  shlo_text) is not None)

  def test_div(self):
    args = (torch.rand((10, 12, 197)), 8.0)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.div.Tensor, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x12x197xf32>.*->.*tensor<\?x12x197xf32>',
                  shlo_text) is not None)

  @unittest.skip("xla::Erf doesn't support unbounded dynamic input.")
  def test_gelu(self):
    args = (torch.rand((3, 5)),)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.gelu, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    # shlo_text = shlo_module.get_stablehlo_text()
    # self.assertTrue(
    #     "(%arg0: tensor<?x2xi64>, %arg1: tensor<?x2xi64>) -> tensor<?x2xi64>" in
    #     shlo_text)

  @unittest.skip("Unbounded Dynamism not supported on view.")
  def test_native_layer_norm(self):
    args = (
        torch.rand((10, 197, 768)),
        [768],
        torch.rand((768)),
        torch.rand((768)),
        1e-12,
    )
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(
        torch.ops.aten.native_layer_norm.default, args, constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>",
                  shlo_text) is not None)

  def test_permute(self):
    args = (torch.rand((10, 197, 12, 64)), [0, 2, 1, 3])
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.permute.default,
                                            args, constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x197x12x64xf32>.*->.*tensor<\?x12x197x64xf32>",
            shlo_text) is not None)

  @unittest.skip("Unbounded Dynamism not supported on select..")
  def test_select(self):
    args = (torch.rand((10, 197, 768)), 1, 0)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.select.int, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x768xf32>",
                  shlo_text) is not None)

  @unittest.skip("Unbounded Dynamism not supported on slice.")
  def test_slice(self):
    args = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten.slice.Tensor, args,
                                            constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x3x224x224xf32>.*->.*tensor<\?x3x224x224xf32>",
            shlo_text) is not None)

  @unittest.skip("Unbounded Dynamism not supported on softmax.")
  def test_softmax(self):
    args = (torch.rand((10, 12, 197, 197)), -1, False)
    constraints = [
        torch.export.dynamic_dim(args[0], 0),
    ]
    ep = self._test_export_dynamism_wrapper(torch.ops.aten._softmax.default,
                                            args, constraints)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x12x197x197xf32>.*->.*tensor<\?x12x197x197xf32>",
            shlo_text) is not None)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
