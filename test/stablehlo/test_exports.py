import unittest
import torch
import torch.nn.functional as F
from torch_xla.stablehlo import exported_program_to_stablehlo


class Interpolate(torch.nn.Module):

  def forward(self, masks: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=(500, 500),
        mode="bilinear",
        align_corners=False,
    )
    return masks


class ExportTest(unittest.TestCase):

  def test_interpolate(self):

    arg = (torch.randn(3, 3, 200, 200),)
    model = Interpolate()

    ans = model(*arg)

    with torch.no_grad():
      exported = torch._export.export(model, arg)
      shlo = exported_program_to_stablehlo(exported)
      ans2 = shlo(*arg).cpu().to(torch.float32)
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))
