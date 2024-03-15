import unittest
import torch
import torch.nn.functional as F
import jax
import torch_xla2
from torch_xla2 import tensor


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
      weights, func = torch_xla2.export.exported_program_to_jax(exported)
      argj = tensor.t2j(arg[0])
      ans2 = jax.jit(func)(weights, (argj,))[0]
      ans2 = tensor.j2t(ans2)
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-3))

  def test_constant(self):

    arg = (torch.randn(10, 10),)
    model = TensorConstant()

    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
      weights, func = torch_xla2.export.exported_program_to_jax(exported)
      argj = tensor.t2j(arg[0])
      ans2 = jax.jit(func)(weights, (argj,))[0]
      ans2 = tensor.j2t(ans2)
      self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))


if __name__ == '__main__':
  unittest.main()
