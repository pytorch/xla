import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import StableHLOExportOptions, exported_program_to_stablehlo
from torch_xla.tf_saved_model_integration import (
    make_tf_function, save_torch_module_as_tf_saved_model,
    save_stablehlo_graph_as_tf)
from torch.utils import _pytree as pytree
from torch.export import export, dynamic_dim
import torch

import tempfile
import unittest
import tensorflow as tf


class StableHLOInferenceTest(unittest.TestCase):

  def test_dynamic_shapes(self):

    class MyModule(torch.nn.Module):

      def forward(self, a, b):
        return a * b

    model = MyModule()
    a = torch.randn(3, 10)
    b = torch.randn(3, 10)
    constraints = [
        dynamic_dim(a, 0),
        dynamic_dim(b, 0),
        dynamic_dim(a, 0) == dynamic_dim(b, 0)
    ]

    exported = torch.export.export(
        model, (
            a,
            b,
        ), constraints=constraints)
    shlo = exported_program_to_stablehlo(exported)
    with tempfile.TemporaryDirectory() as tempdir:
      save_stablehlo_graph_as_tf(
          shlo,
          tempdir,
          serving_key=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
          function_alias='')
      loaded_m = tf.saved_model.load(tempdir)
      data2 = (np.random.randn(2, 10), np.random.randn(2, 10))
      res = loaded_m.f(*data2)
      self.assertTrue(np.allclose(res, data2[0] * data2[1]))

  def test_unused_param(self):

    class M(torch.nn.Module):

      def forward(self, a, b):
        return torch.sin(b)

    model = M()
    data = (torch.randn(4, 3, 224, 224), torch.randn(1, 100))
    output = model(*data)

    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_module_as_tf_saved_model(model, data, tempdir)
      loaded_m = tf.saved_model.load(tempdir)
      res = loaded_m.f(data[0].detach().numpy(), data[1].detach().numpy())[0]
      output2 = torch.tensor(res.numpy())
      self.assertTrue(torch.allclose(output, output2, atol=1e-5))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
