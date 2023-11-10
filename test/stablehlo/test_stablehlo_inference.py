import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo, StableHLOExportOptions
from torch.utils import _pytree as pytree
import torch
import torchvision

import tempfile
import unittest


def export_torch_model(model, args):
  exported = torch._export.export(model, args)
  options = StableHLOExportOptions()
  options.override_tracing_arguments = args
  return exported_program_to_stablehlo(exported, options)


class StableHLOInferenceTest(unittest.TestCase):

  def test_resnet18_inference(self):
    resnet18 = torchvision.models.resnet18()
    data = torch.randn(4, 3, 224, 224)
    output = resnet18(data)

    exported = export_torch_model(
        resnet18,
        (data,),
    )
    output2 = exported(data).cpu()
    self.assertTrue(torch.allclose(output, output2, atol=1e-3))

  def test_resnet18_save_load(self):
    return
    import tensorflow as tf
    resnet18 = torchvision.models.resnet18()
    data = torch.randn(4, 3, 224, 224)
    output = resnet18(data)
    output_np = output.detach().numpy()

    exported = export_torch_model(resnet18, (data,))
    tf_m = tf.Module()
    tf_m.f = tf.function(
        exported,
        autograph=False,
        input_signature=[tf.TensorSpec([4, 3, 224, 224], tf.float32)])
    with tempfile.TemporaryDirectory() as tempdir:
      tf.saved_model.save(tf_m, tempdir)
      loaded_m = tf.saved_model.load(tempdir)
      res = loaded_m.f(data.detach().numpy())[0]
      output2 = torch.tensor(res.numpy())
      self.assertTrue(torch.allclose(output, output2, atol=1e-5))

  def test_add_float(self):

    class AddFloat(torch.nn.Module):

      def __init__(self) -> None:
        super().__init__()

      # def forward(self, tensors, dim=0):
      def forward(self, x, y):
        return x + y

    m = AddFloat()
    data = (torch.randn(100, 100), torch.tensor(4.4))
    output = m(*data)
    exported = export_torch_model(m, data)

    device = xm.xla_device()
    data = pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), data)
    output2 = exported(*data).cpu()

    self.assertTrue(torch.allclose(output, output2, atol=1e-3))

  def test_model_with_dict(self):

    class DictInput(torch.nn.Module):

      def __init__(self) -> None:
        super().__init__()

      def forward(self, inputs):
        return (inputs['x'] + inputs['y'], inputs['x'] * inputs['y'])

    m = DictInput()
    data = ({
        'x': torch.randn((10, 10)),
        'y': torch.randn((10, 10)),
    },)

    output = m(*data)
    exported = export_torch_model(m, data)
    device = xm.xla_device()
    data = pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), data)
    output2 = exported(*data)
    self.assertEqual(len(output2), 2)

    self.assertTrue(torch.allclose(output[0], output2[0].cpu()))
    self.assertTrue(torch.allclose(output[1], output2[1].cpu()))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
