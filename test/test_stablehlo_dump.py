import torch_xla
import torch_xla.core.xla_model as xm
import torch
import torchvision
import unittest


class StableHloDumpTest(unittest.TestCase):

  def test_simple(self):
    device = xm.xla_device()
    x = torch.tensor([3], device=device)
    y = torch.tensor([3], device=device)
    z = x + y
    # Example usage of dumping StableHLO given output tensors
    stablehlo = xm.get_stablehlo([z])
    self.assertEqual(stablehlo.count("stablehlo.multiply"), 1)
    self.assertEqual(stablehlo.count("stablehlo.add"), 1)

  def test_resnet18(self):
    device = xm.xla_device()
    xla_resnet18 = torchvision.models.resnet18()
    xla_resnet18.eval()
    xla_resnet18 = xla_resnet18.to(device)
    data = torch.randn(4, 3, 224, 224, device=device)
    output = xla_resnet18(data)
    stablehlo = xm.get_stablehlo()
    self.assertEqual(stablehlo.count("convolution"), 20)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
