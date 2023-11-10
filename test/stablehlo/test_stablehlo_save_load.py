import tempfile
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import save_torch_model_as_stablehlo, save_as_stablehlo
from torch_xla.stablehlo import StableHLOExportOptions, StableHLOGraphModule
import torch
import torch._export
import torchvision
import unittest
from torch import nn

from typing import Tuple

Tensor = torch.Tensor


class StableHloDumpTest(unittest.TestCase):

  def test_simple(self):
    device = xm.xla_device()
    x = torch.tensor([3], device=device)
    y = torch.tensor([3], device=device)
    z = x + y
    # Example usage of dumping StableHLO given output tensors
    stablehlo = xm.get_stablehlo([z])
    self.assertEqual(stablehlo.count("stablehlo.add"), 1)

  def test_resnet18(self):
    device = xm.xla_device()
    xla_resnet18 = torchvision.models.resnet18()
    xla_resnet18.eval()
    xla_resnet18 = xla_resnet18.to(device)
    data = torch.randn(4, 3, 224, 224, device=device)
    output = xla_resnet18(data)
    stablehlo = xm.get_stablehlo([output])
    self.assertEqual(stablehlo.count("stablehlo.convolution"), 20)
    stablehlo = xm.get_stablehlo()
    self.assertEqual(stablehlo.count("stablehlo.convolution"), 20)
    self.assertGreater(stablehlo.count("#loc"), 0)


class ElementwiseAdd(nn.Module):

  def __init__(self) -> None:
    super().__init__()

  def forward(self, x: Tensor, y: Tensor) -> Tensor:
    return x + y

  def get_random_inputs(self) -> Tuple[Tensor, Tensor]:
    return (torch.randn(1, 3), torch.randn(1, 3))


class Cat(nn.Module):

  def __init__(self) -> None:
    super().__init__()

  # def forward(self, tensors, dim=0):
  def forward(self, *args: Tensor, dim: int) -> Tensor:
    return torch.cat(args, dim)


class SimpleExportTest(unittest.TestCase):

  def export_stable_hlo(self, model, args, kwargs=None):
    if kwargs is None:
      kwargs = {}
    device = xm.xla_device()
    model.eval()
    model = model.to(device)
    args = tuple(i.to(device) for i in args if hasattr(i, 'to'))
    output = model(*args, **kwargs)
    if not isinstance(output, tuple):
      output = [output]
    return xm.get_stablehlo(output)

  def test_add(self):
    model = ElementwiseAdd()
    inputs = model.get_random_inputs()
    self.export_stable_hlo(model, inputs)  # works

  def test_cat(self):
    model = Cat()
    inputs = (torch.randn(10, 10), torch.randn(10, 10))
    kwargs = {'dim': 1}
    stablehlo = self.export_stable_hlo(model, inputs, kwargs)
    # FIXME: Currently the dim=1 is hard coded
    self.assertTrue('dim = 1' in stablehlo)

  def test_save_load(self):
    model = ElementwiseAdd()
    inputs = model.get_random_inputs()
    exported = torch._export.export(model, inputs)
    options = StableHLOExportOptions()
    options.override_tracing_arguments = inputs
    with tempfile.TemporaryDirectory() as tempdir:
      save_as_stablehlo(exported, tempdir, options)
      program2 = StableHLOGraphModule.load(tempdir)
    result = program2(*inputs).detach().cpu()
    self.assertTrue(torch.allclose(model(*inputs), result))

  def test_save_load2(self):
    model = ElementwiseAdd()
    inputs = model.get_random_inputs()
    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_model_as_stablehlo(model, inputs, tempdir)
      program2 = StableHLOGraphModule.load(tempdir)
      result = program2(*inputs).detach().cpu()
    self.assertTrue(torch.allclose(model(*inputs), result))

  def test_save_load3(self):
    model = ElementwiseAdd()
    inputs = model.get_random_inputs()
    exported = torch._export.export(model, inputs)
    with tempfile.TemporaryDirectory() as tempdir:
      # Shouldnt need specify options because exported has example_input inside
      save_as_stablehlo(exported, tempdir)
      program2 = StableHLOGraphModule.load(tempdir)
    result = program2(*inputs).detach().cpu()
    self.assertTrue(torch.allclose(model(*inputs), result))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
