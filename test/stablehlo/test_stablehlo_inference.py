import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.stablehlo_saved_model import export_torch_model
import torch
import torchvision

import tempfile
import unittest


class StableHLOInferenceTest(unittest.TestCase):

  def test_resnet18_inference(self):
    resnet18 = torchvision.models.resnet18()
    data = torch.randn(4, 3, 224, 224)
    output = resnet18(data)

    exported = export_torch_model(
        resnet18,
        (data,),
    )
    output2 = exported(data)[0].cpu()

    self.assertTrue(torch.allclose(output, output2, atol=1e-5))

  def test_resnet18_save_load(self):
    import tensorflow as tf
    resnet18 = torchvision.models.resnet18()
    data = torch.randn(4, 3, 224, 224)
    output = resnet18(data)
    output_np = output.detach().numpy()

    exported = export_torch_model(
        resnet18,
        (data,),
        to_tf=True
    )
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
    output2 = exported(*data)[0].cpu()

    self.assertTrue(torch.allclose(output, output2, atol=1e-5))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
