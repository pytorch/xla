import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import StableHLOExportOptions, exported_program_to_stablehlo
from torch_xla.tf_saved_model_integration import make_tf_function, save_torch_module_as_tf_saved_model
from torch.utils import _pytree as pytree
import torch
import torchvision

import tempfile
import unittest
import tensorflow as tf


class StableHLOInferenceTest(unittest.TestCase):

  def test_resnet18_inference(self):
    resnet18 = torchvision.models.resnet18().eval()
    data = torch.randn(4, 3, 224, 224)
    output = resnet18(data)

    exported = torch.export.export(resnet18, (data,))
    options = StableHLOExportOptions(override_tracing_arguments=(data,))
    stablehlo_program = exported_program_to_stablehlo(exported, options)
    tf_func = make_tf_function(stablehlo_program)

    output_tf = tf_func(*options.override_tracing_arguments)
    output2 = torch.tensor(output_tf[0].numpy())
    self.assertTrue(torch.allclose(output, output2, atol=1e-5))

  def test_resnet18_save_load(self):
    resnet18 = torchvision.models.resnet18().eval()
    data = torch.randn(4, 3, 224, 224)
    output = resnet18(data)

    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_module_as_tf_saved_model(resnet18, (data,), tempdir)
      loaded_m = tf.saved_model.load(tempdir)
      res = loaded_m.f(data.detach().numpy())[0]
      output2 = torch.tensor(res.numpy())
      self.assertTrue(torch.allclose(output, output2, atol=1e-5))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
