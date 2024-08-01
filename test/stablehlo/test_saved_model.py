import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch.export import Dim, export
from torch.utils import _pytree as pytree
from torch_xla.stablehlo import (StableHLOExportOptions,
                                 exported_program_to_stablehlo)
from torch_xla.tf_saved_model_integration import (
    make_tf_function, save_stablehlo_graph_as_tf,
    save_torch_module_as_tf_saved_model)
from torch_xla.utils.stablehlo_test_utils import (
    compare_exported_program_and_saved_model_result, has_tf_package,
    wrap_func_as_nn_module)


class StableHLOInferenceTest(unittest.TestCase):

  def test_dynamic_shapes(self):

    class MyModule(torch.nn.Module):

      def forward(self, a, b):
        return a * b

    model = MyModule()
    a = torch.randn(3, 10)
    b = torch.randn(3, 10)
    bs = Dim("bs")
    dynamic_shapes = ({0: bs}, {0: bs})

    exported = torch.export.export(
        model, (
            a,
            b,
        ), dynamic_shapes=dynamic_shapes)
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

    m = M()
    args = (torch.randn(4, 3, 224, 224), torch.randn(1, 100))
    ep = torch.export.export(m, args)
    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_module_as_tf_saved_model(m, args, tempdir)
      self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
      compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_multiple_outputs(self):

    class M(torch.nn.Module):

      def forward(self, a, b):
        return a + b, a * b, a, b

    m = M()
    args = (torch.rand((2, 3)), torch.rand((2, 3)))
    ep = torch.export.export(m, args)
    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_module_as_tf_saved_model(m, args, tempdir)
      self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
      compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_non_tensor_input_int(self):
    m = wrap_func_as_nn_module(torch.ops.aten._softmax.default)
    args = (torch.rand((2, 3, 4, 5)), -1, False)
    ep = torch.export.export(m, args)
    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_module_as_tf_saved_model(m, args, tempdir)
      self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
      compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_non_tensor_input_float(self):
    m = wrap_func_as_nn_module(torch.ops.aten._cdist_forward)
    args = (torch.rand((2, 3, 4)), torch.rand((2, 3, 4)), 2.4, 1)
    ep = torch.export.export(m, args)
    with tempfile.TemporaryDirectory() as tempdir:
      save_torch_module_as_tf_saved_model(m, args, tempdir)
      self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
      compare_exported_program_and_saved_model_result(ep, tempdir, args)


if __name__ == '__main__':
  if not has_tf_package():
    print("skip tf.saved_model tests, tf is not installed.")
    sys.exit(0)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
