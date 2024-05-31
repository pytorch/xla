import os
import re
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from torch.export import Dim, export
from torch_xla.stablehlo import exported_program_to_stablehlo

try:
  from torch_xla.tf_saved_model_integration import \
      save_torch_module_as_tf_saved_model
except ImportError:
  print("tf is not installed. The tf.saved_model tests will be skipped.")
from torch_xla.utils.stablehlo_test_utils import (
    compare_exported_program_and_saved_model_result, has_tf_package,
    load_save_model_and_inference, wrap_func_as_nn_module)

device = xm.xla_device()
os.environ['EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM'] = '1'


class UnboundedDynamismExportTest(unittest.TestCase):

  def test_add(self):
    args = (torch.rand((10, 197, 768)), torch.rand((10, 197, 768)))
    dynamic_shapes = (({0: Dim("dim")}, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.add.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<\?x197x768xf32>.*tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_add_scalar(self):
    args = (torch.rand((10, 197, 768)), 0.345)
    dynamic_shapes = (({0: Dim("dim")}, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.add.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_addmm(self):
    args = (torch.rand((5)), torch.rand((10, 5)), torch.rand((5, 5)))
    dynamic_shapes = ((None, {0: Dim("dim")}, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.addmm.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x5xf32>.*->.*tensor<\?x5xf32>', shlo_text)
        is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        # Hit stablehlo.dot shape refinement error when inferencing saved_model in TF.
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_bmm(self):
    args = (
        torch.rand((24, 197, 64)),
        torch.rand((24, 64, 197)),
    )
    dynamic_shapes = (({0: Dim("dim")}, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.bmm.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<\?x64x197xf32>.*%arg.: tensor<\?x197x64xf32>.*->.*tensor<\?x197x197xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_bmm_dynamic_out_dim(self):
    args = (
        torch.rand((8, 128, 256)),
        torch.rand((8, 256, 3)),
    )
    dynamic_shapes = ((None, {2: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.bmm.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<8x256x\?xf32>.*%arg.: tensor<8x128x256xf32>.*->.*tensor<8x128x\?xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_bmm_dynamic_reduction_dim(self):
    args = (
        torch.rand((8, 128, 3)),
        torch.rand((8, 3, 256)),
    )
    dynamic_shapes = (({2: Dim("dim")}, {1: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.bmm.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<8x\?x256xf32>.*%arg.: tensor<8x128x\?xf32>.*->.*tensor<8x128x256xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_cat(self):
    args = (torch.rand((10, 1, 768)), torch.rand((10, 196, 768)))
    dynamic_shapes = (({0: Dim("dim")}, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(
        lambda x, y: torch.ops.aten.cat.default([x, y], 1))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'%arg.: tensor<\?x196x768xf32>.*%arg.: tensor<\?x1x768xf32>.*->.*tensor<\?x197x768xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_conv(self):
    args = (
        torch.rand((10, 3, 224, 224)),
        torch.rand((5, 3, 16, 16)),
        torch.rand((5)),
    )
    dynamic_shapes = (({0: Dim("dim")}, None, None),)
    m = wrap_func_as_nn_module(
        lambda x, y, z: torch.ops.aten.convolution.default(
            x,
            y,
            z,
            [16, 16],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        ))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x3x224x224xf32>.*->.*tensor<\?x5x14x14xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_conv1d(self):
    args = (
        torch.rand((3, 1, 800)),
        torch.rand((512, 1, 10)),
    )
    dynamic_shapes = (({0: Dim("dim")}, None),)
    # dynamic_shapes = None
    m = wrap_func_as_nn_module(lambda x, y: torch.ops.aten.convolution.default(
        x,
        y,
        None,
        [5],
        [0],
        [1],
        False,
        [0],
        1,
    ))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x1x800xf32>.*->.*tensor<\?x512x159xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_cumsum(self):
    args = (torch.rand((10, 5)), 1)
    dynamic_shapes = (({0: Dim("dim")}, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.cumsum.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x5xf32>.*->.*tensor<\?x5xf32>', shlo_text)
        is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_div(self):
    args = (torch.rand((10, 12, 197)), torch.rand((10, 12, 197)))
    dynamic_shapes = (({0: Dim("dim")}, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.div.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<\?x12x197xf32>.*tensor<\?x12x197xf32>.*->.*tensor<\?x12x197xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_div_scalar(self):
    args = (torch.rand((10, 12, 197)), 8.0)
    dynamic_shapes = (({0: Dim("dim")}, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.div.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x12x197xf32>.*->.*tensor<\?x12x197xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_gelu(self):
    args = (torch.rand((3, 5)),)
    dynamic_shapes = (({0: Dim("dim")},),)
    m = wrap_func_as_nn_module(torch.ops.aten.gelu)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x5xf32>.*->.*tensor<\?x5xf32>', shlo_text)
        is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_embedding(self):

    class M(torch.nn.Module):

      def forward(self, x, y):
        res = torch.ops.aten.embedding.default(x, y)
        return res

    args = (torch.rand((20, 768)), torch.randint(0, 15,
                                                 (3, 10)).to(torch.int64))
    dynamic_shapes = (None, {0: Dim("dim")})
    m = M()
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x10xi64>.*->.*tensor<\?x10x768xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_mean(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return torch.mean(x, -1, keepdim=True)

    args = (torch.rand((10, 197, 768)),)
    dynamic_shapes = ({0: Dim("dim")},)
    m = M()
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x197x1xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_mul(self):
    args = (torch.rand((10, 2, 768)), torch.rand((10, 2, 768)))
    dynamic_shapes = (({0: Dim("dim")}, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.mul.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<\?x2x768xf32>.*tensor<\?x2x768xf32>.*->.*tensor<\?x2x768xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_mul_scalar(self):
    args = (torch.rand((10, 2, 768)), 0.125)
    dynamic_shapes = (({0: Dim("dim")}, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.mul.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x2x768xf32>.*->.*tensor<\?x2x768xf32>', shlo_text)
        is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_ne_scalar(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.ne.Scalar(x, 1).to(torch.int32)

    args = (torch.rand((3, 5)).to(torch.int64),)
    dynamic_shapes = ({0: Dim("dim")},)
    m = M()
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x5xi64>.*->.*tensor<\?x5xi32>", shlo_text)
        is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_var(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return torch.var(x, -1, keepdim=True, correction=0)

    args = (torch.rand((10, 197, 768)),)
    dynamic_shapes = ({0: Dim("dim")},)
    m = M()
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x197x1xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_native_group_norm(self):

    class M2(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.GroupNorm(
            num_groups=512, num_channels=512, affine=True)

      def forward(self, x):
        x = self.layer_norm(x)
        return x

    args = (torch.rand((10, 512, 159)),)
    dynamic_shapes = ({0: Dim("dim")},)
    m = M2()
    out1 = m(*args)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x512x159xf32>.*->.*tensor<\?x512x159xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(
            ep, tempdir, args, atol=1e-6)

  def test_native_layer_norm(self):

    class M(torch.nn.Module):

      def forward(self, x, weight, bias):
        return torch.ops.aten.native_layer_norm.default(x, [768], weight, bias,
                                                        1e-12)[0]

    args = (
        torch.rand((10, 197, 768)),
        torch.rand((768)),
        torch.rand((768)),
    )
    dynamic_shapes = ({0: Dim("dim")}, None, None)
    m = M()
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x197x768xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(
            ep, tempdir, args, atol=1e-6)

  def test_permute(self):
    args = (torch.rand((10, 197, 12, 64)),)
    dynamic_shapes = (({0: Dim("dim")},),)
    m = wrap_func_as_nn_module(
        lambda x: torch.ops.aten.permute.default(x, [0, 2, 1, 3]))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x197x12x64xf32>.*->.*tensor<\?x12x197x64xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_select(self):
    args = (torch.rand((10, 197, 768)), 1, 0)
    dynamic_shapes = (({0: Dim("dim")}, None, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.select.int)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x768xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_slice(self):
    args = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
    dynamic_shapes = (({0: Dim("dim")}, None, None, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.slice.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x3x224x224xf32>.*->.*tensor<\?x3x224x224xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_slice_2(self):
    args = (torch.rand((10, 3, 224, 224)), 1, 0, 2)
    dynamic_shapes = (({0: Dim("dim")}, None, None, None),)
    m = wrap_func_as_nn_module(torch.ops.aten.slice.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x3x224x224xf32>.*->.*tensor<\?x2x224x224xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_softmax(self):
    args = (torch.rand((10, 12, 197, 197)), -1, False)
    dynamic_shapes = (({0: Dim("dim")}, None, None),)
    m = wrap_func_as_nn_module(torch.ops.aten._softmax.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x12x197x197xf32>.*->.*tensor<\?x12x197x197xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_sub(self):
    args = (torch.rand((10, 1, 1, 10)), torch.rand((10, 1, 1, 10)))
    dynamic_shapes = (({0: Dim("dim")}, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.sub.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r'tensor<\?x1x1x10xf32>.*tensor<\?x1x1x10xf32>.*->.*tensor<\?x1x1x10xf32>',
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_softmax_reduce_on_dynamic_dim(self):
    args = (torch.rand((1, 8, 128, 3)), -1, False)
    dynamic_shapes = (({3: Dim("dim")}, None, None),)
    m = wrap_func_as_nn_module(torch.ops.aten._softmax.default)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<1x8x128x\?xf32>.*->.*tensor<1x8x128x\?xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  @unittest.skip("Converted StableHLO contains i1 dtype, not expected.")
  def test_index(self):
    args = (torch.rand((2, 10)), torch.arange(5))
    dynamic_shapes = ((None, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(
        lambda x, y: torch.ops.aten.index.Tensor(x, [None, y]))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?xi64>.*%arg.: tensor<2x10xf32>.*->.*tensor<2x\?xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_sub_scalar(self):
    args = (1.0, torch.rand((10, 1, 1, 10)))
    dynamic_shapes = ((None, {0: Dim("dim")}),)
    m = wrap_func_as_nn_module(torch.ops.aten.sub.Tensor)
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r'tensor<\?x1x1x10xf32>.*->.*tensor<\?x1x1x10xf32>',
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_split_with_sizes(self):

    class M(torch.nn.Module):

      def forward(self, x):
        res = torch.ops.aten.split_with_sizes.default(x, [1, 2, 3], -1)
        return res[0], res[1], res[2]

    args = (torch.rand((3, 10, 6)),)
    dynamic_shapes = ({0: Dim("dim")},)
    m = M()
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x10x6xf32>.*->.*tensor<\?x10x1xf32>.*tensor<\?x10x2xf32>.*tensor<\?x10x3xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_transpose_on_dynamic_dim(self):
    args = (torch.rand((1, 8, 3, 256)),)
    dynamic_shapes = (({2: Dim("dim")},),)
    m = wrap_func_as_nn_module(
        lambda x: torch.ops.aten.transpose.int(x, -2, -1))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<1x8x\?x256xf32>.*->.*tensor<1x8x256x\?xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_unsqueeze_1(self):
    args = (torch.rand((3, 10)),)
    dynamic_shapes = (({0: Dim("dim")},),)
    m = wrap_func_as_nn_module(lambda x: torch.ops.aten.unsqueeze.default(x, 1))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x10xf32>.*->.*tensor<\?x1x10xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_unsqueeze_2(self):
    args = (torch.rand((1, 1, 3, 256)),)
    dynamic_shapes = (({2: Dim("dim")},),)
    m = wrap_func_as_nn_module(lambda x: torch.ops.aten.unsqueeze.default(x, 2))
    ep = export(m, args=args, dynamic_shapes=dynamic_shapes)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<1x1x\?x256xf32>.*->.*tensor<1x1x1x\?x256xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        compare_exported_program_and_saved_model_result(ep, tempdir, args)

  def test_dynamic_view(self):

    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, [16, 16])

      def forward(self, x):
        x = self.conv(x)
        return x.view(x.shape[0], x.shape[1], -1)

    m = M().eval()
    args = (torch.rand((10, 3, 224, 224)),)
    dynamic_shapes = ({0: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep.module()(*args)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x3x224x224xf32>.*->.*tensor<\?x5x43681xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        tf_out = load_save_model_and_inference(tempdir, args)
        self.assertTrue(
            np.allclose(out1.detach().numpy(), tf_out[0].numpy(), atol=1e-05))

  @unittest.skip("Cannot generate aten.sym_numel in the exported program.")
  def test_dynamic_view_sym_numel(self):

    class M(torch.nn.Module):

      def forward(self, x, range):
        num_elem = torch.numel(range)
        return x.view(x.shape[0], x.shape[2], num_elem, x.shape[4])

    m = M().eval()
    args = (torch.rand((1, 1, 8, 3, 256)), torch.arange(3))
    dynamic_shapes = ({3: Dim("bs")}, {0: Dim("bs")})
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    print(ep)
    out1 = ep.module()(*args)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<\?x3x224x224xf32>.*->.*tensor<\?x5x43681xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        tf_out = load_save_model_and_inference(tempdir, args)
        self.assertTrue(
            np.allclose(out1.detach().numpy(), tf_out[0].numpy(), atol=1e-05))

  def test_dynamic_view_non_bs(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

    m = M().eval()
    args = (torch.rand((1, 3, 2, 16)),)
    dynamic_shapes = ({1: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep.module()(*args)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<1x\?x2x16xf32>.*->.*tensor<1x\?x16xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        tf_out = load_save_model_and_inference(tempdir, args)
        self.assertTrue(
            np.allclose(out1.detach().numpy(), tf_out[0].numpy(), atol=1e-05))

  def test_dynamic_view_multiplier(self):

    class M(torch.nn.Module):

      def forward(self, x):
        return x.view(x.shape[0] * x.shape[1], -1)

    m = M().eval()
    args = (torch.rand((10, 197, 768)),)
    dynamic_shapes = ({0: Dim("bs")},)
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep.module()(*args)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<\?x197x768xf32>.*->.*tensor<\?x768xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        tf_out = load_save_model_and_inference(tempdir, args)
        self.assertTrue(
            np.allclose(out1.detach().numpy(), tf_out[0].numpy(), atol=1e-05))

  def test_dynamic_expand(self):

    class M(torch.nn.Module):

      def forward(self, x, image):
        return x.expand(image.shape[0], -1, -1)

    m = M().eval()
    args = (torch.rand((1, 1, 768)), torch.rand((10, 3, 224, 224)))
    dynamic_shapes = (
        None,
        {
            0: Dim("bs")
        },
    )
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep.module()(*args)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(r"%arg.: tensor<1x1x768xf32>.*->.*tensor<\?x1x768xf32>",
                  shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        tf_out = load_save_model_and_inference(tempdir, args)
        self.assertTrue(
            np.allclose(out1.detach().numpy(), tf_out[0].numpy(), atol=1e-05))

  def test_dynamic_expand_2(self):

    class M(torch.nn.Module):

      def forward(self, x, range):
        return x.expand(1, 1, 8, range.shape[0], 256)

    m = M().eval()
    args = (torch.rand((1, 1, 1, 3, 256)), torch.arange(3))
    dynamic_shapes = ({3: Dim("bs")}, {0: Dim("bs")})
    ep = export(m, args, dynamic_shapes=dynamic_shapes)
    out1 = ep.module()(*args)
    shlo_module = exported_program_to_stablehlo(ep)
    shlo_text = shlo_module.get_stablehlo_text()
    self.assertTrue(
        re.search(
            r"%arg.: tensor<1x1x1x\?x256xf32>.*->.*tensor<1x1x8x\?x256xf32>",
            shlo_text) is not None)
    if has_tf_package():
      with tempfile.TemporaryDirectory() as tempdir:
        save_torch_module_as_tf_saved_model(
            m, args, tempdir, dynamic_shapes=dynamic_shapes)
        self.assertTrue(os.path.exists(os.path.join(tempdir, 'saved_model.pb')))
        tf_out = load_save_model_and_inference(tempdir, args)
        self.assertTrue(
            np.allclose(out1.detach().numpy(), tf_out[0].numpy(), atol=1e-05))


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
