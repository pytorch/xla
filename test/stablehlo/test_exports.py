import re
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch_xla.stablehlo import (StableHLOExportOptions, StableHLOGraphModule,
                                 exported_program_to_stablehlo)


class Interpolate(torch.nn.Module):

  def forward(self, masks: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=(500, 500),
        mode="bilinear",
        align_corners=False,
    )
    return masks


class TensorConstant(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a):
    return a / torch.tensor(3)


class ExportTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)

  def test_interpolate(self):

    arg = (torch.randn(3, 3, 200, 200),)
    model = Interpolate()

    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
      shlo = exported_program_to_stablehlo(exported)
      ans2 = shlo(*arg).cpu().to(torch.float32)
      torch.testing.assert_close(ans, ans2, rtol=1e-5, atol=1e-4)

  def test_constant(self):

    arg = (torch.randn(10, 10),)
    model = TensorConstant()

    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
      shlo = exported_program_to_stablehlo(exported)
      ans2 = shlo(*arg).cpu().to(torch.float32)
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))

      with tempfile.TemporaryDirectory() as tempdir:
        # Shouldnt need specify options because exported has example_input inside
        shlo.save(tempdir)
        program2 = StableHLOGraphModule.load(tempdir)
      result = program2(*arg).detach().cpu()
      self.assertTrue(torch.allclose(ans, result))

  def test_inline_all_scalar(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.sub(torch.ops.aten.sub(x, 5), 1)

    arg = (torch.randn(10, 10),)
    model = M()

    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
      shlo = exported_program_to_stablehlo(exported)
      ans2 = shlo(*arg).cpu().to(torch.float32)
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))
      shlo_text = shlo.get_stablehlo_text()
      self.assertTrue('stablehlo.constant dense<1.000000e+00>' in shlo_text)
      self.assertTrue('stablehlo.constant dense<5.000000e+00>' in shlo_text)
      self.assertFalse(re.search(r'%arg.*tensor<f32>', shlo_text) is not None)

  def test_inline_only_special_scalar(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.sub(torch.ops.aten.sub(x, 5), 1)

    arg = (torch.randn(10, 10),)
    model = M()

    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
      export_options = StableHLOExportOptions()
      export_options.inline_all_constant = False
      shlo = exported_program_to_stablehlo(exported, options=export_options)
      ans2 = shlo(*arg).cpu().to(torch.float32)
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))
      shlo_text = shlo.get_stablehlo_text()
      self.assertTrue('stablehlo.constant dense<1.000000e+00>' in shlo_text)
      self.assertTrue(re.search(r'%arg.*tensor<f32>', shlo_text) is not None)
      self.assertFalse('stablehlo.constant dense<5.000000e+00>' in shlo_text)

  def test_export_no_weights(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))

      def forward(self, x):
        return torch.ops.aten.add(x, self.weight)

    arg = (torch.randn(10, 10),)
    model = M()
    with torch.no_grad():
      exported = torch.export.export(model, arg)
      export_options = StableHLOExportOptions()
      export_options.export_weights = False
      shlo = exported_program_to_stablehlo(exported, options=export_options)
      self.assertEqual(shlo._bundle.state_dict, {})


if __name__ == '__main__':
  unittest.main()
