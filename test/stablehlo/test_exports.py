import tempfile
import unittest
import torch
import torch.nn.functional as F
from torch_xla.stablehlo import exported_program_to_stablehlo, StableHLOGraphModule


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
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))

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


if __name__ == '__main__':
  unittest.main()
