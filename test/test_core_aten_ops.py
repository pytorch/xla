import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla.stablehlo import exported_program_to_stablehlo
from torch.utils import _pytree as pytree
import torch

import os
import tempfile
import unittest


def diff_output(testcase, output1, output2, rtol, atol, equal_nan=True):
  if isinstance(output1, torch.Tensor):
    testcase.assertIsInstance(output2, torch.Tensor)
    output2_cpu = output2.detach().cpu()
    if output2_cpu.dtype != output1.dtype:
      output2_cpu = output2_cpu.to(output1.dtype)
    testcase.assertEqual(output1.shape, output2.shape)
    testcase.assertTrue(
        torch.allclose(
            output1, output2_cpu, atol=atol, rtol=rtol, equal_nan=equal_nan))
  elif isinstance(output1, (tuple, list)):
    testcase.assertIsInstance(output2, (tuple, list))
    testcase.assertEqual(len(output1), len(output2))
    for o1, o2 in zip(output1, output2):
      diff_output(testcase, o1, o2, rtol, atol)
  else:
    testcase.assertEqual(output1, output2)


class NNModWrapper(torch.nn.Module):

  def __init__(self, op):
    super().__init__()
    self._op = op

  def forward(self, *args, **kwargs):
    return self._op(*args, **kwargs)


def run_export_and_compare(testcase,
                           func,
                           args,
                           kwargs,
                           atol=1e-3,
                           rtol=1e-5,
                           equal_nan=True):
  device = xm.xla_device()
  with testcase.subTest('torch_eval'):
    res = func(*args, **kwargs)
    with testcase.subTest('torch_xla_eval'):
      args2 = pytree.tree_map_only(torch.Tensor, lambda x: x.to(device=device),
                                   args)
      kwargs2 = pytree.tree_map_only(torch.Tensor,
                                     lambda x: x.to(device=device), kwargs)
      res_xla = func(*args2, **kwargs2)
      with testcase.subTest('torch_xla_metric'):
        aten_function_name = f'aten::{str(func).split(".")[-1]}'
        testcase.assertNotIn(aten_function_name, met.metrics_report())
      with testcase.subTest('torch_xla_diff:' + str(atol)):
        diff_output(
            testcase, res, res_xla, atol=atol, rtol=rtol, equal_nan=equal_nan)
    with testcase.subTest('can_export'):
      exported = torch.export.export(NNModWrapper(func), args, kwargs)
      with testcase.subTest('can_convert_to_stablehlo'):
        shlo = exported_program_to_stablehlo(exported)
        with testcase.subTest('stablehlo_can_run'):
          res2 = shlo(*args, **kwargs)
          with testcase.subTest('stablehlo_diff: ' + str(atol)):
            diff_output(
                testcase, res, res2, rtol=rtol, atol=atol, equal_nan=equal_nan)


class AtenOpTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    met.clear_all()

  def test_aten_abs_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.abs, args, kwargs)

  def test_aten_abs_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.abs, args, kwargs)

  def test_aten_abs_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.abs, args, kwargs)

  def test_aten_acos_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.acos, args, kwargs)

  def test_aten_acos_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.acos, args, kwargs)

  def test_aten_acos_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.acos, args, kwargs)

  def test_aten_acosh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.acosh, args, kwargs)

  def test_aten_acosh_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.acosh, args, kwargs)

  def test_aten_acosh_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.acosh, args, kwargs)

  def test_aten_unsqueeze_0(self):
    args = (
        torch.randn((1, 3, 10)).to(torch.float32),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_1(self):
    args = (
        torch.randn((1, 3, 10)).to(torch.float16),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_2(self):
    args = (
        torch.randint(0, 10, (1, 3, 10)).to(torch.int32),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_3(self):
    args = (
        torch.randn((1, 3, 10)).to(torch.float32),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_4(self):
    args = (
        torch.randn((1, 3, 10)).to(torch.float16),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_5(self):
    args = (
        torch.randint(0, 10, (1, 3, 10)).to(torch.int32),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_6(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_7(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten_unsqueeze_8(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze, args, kwargs)

  def test_aten__adaptive_avg_pool2d_0(self):
    args = (
        torch.randn((1, 3, 1, 10)).to(torch.float32),
        [
            1,
            5,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._adaptive_avg_pool2d, args,
                           kwargs)

  def test_aten__adaptive_avg_pool2d_1(self):
    args = (
        torch.randn((1, 3, 10, 10)).to(torch.float32),
        [
            5,
            5,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._adaptive_avg_pool2d, args,
                           kwargs)

  def test_aten_squeeze_dim_0(self):
    args = (
        torch.randn((1, 3, 1, 5)).to(torch.float32),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dim, args, kwargs)

  def test_aten_squeeze_dim_1(self):
    args = (
        torch.randn((1, 3, 1, 5)).to(torch.float32),
        -2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dim, args, kwargs)

  def test_aten_squeeze_dim_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dim, args, kwargs)

  def test_aten_squeeze_dim_3(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dim, args, kwargs)

  def test_aten_squeeze_dim_4(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dim, args, kwargs)

  def test_aten__adaptive_avg_pool3d_0(self):
    args = (
        torch.randn((1, 3, 10, 10, 10)).to(torch.float32),
        [
            5,
            5,
            5,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._adaptive_avg_pool3d, args,
                           kwargs)

  def test_aten__adaptive_avg_pool3d_1(self):
    args = (
        torch.randn((1, 3, 10, 10, 10)).to(torch.float16),
        [
            5,
            5,
            5,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._adaptive_avg_pool3d, args,
                           kwargs)

  def test_aten_add_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.add.Scalar, args, kwargs)

  def test_aten_add_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.add.Scalar, args, kwargs)

  def test_aten_add_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.add.Scalar, args, kwargs)

  def test_aten_add_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.add.Tensor, args, kwargs)

  def test_aten_add_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.add.Tensor, args, kwargs)

  def test_aten_add_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.add.Tensor, args, kwargs)

  def test_aten_addmm_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.addmm, args, kwargs)

  def test_aten_addmm_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.addmm, args, kwargs, atol=1e-2)

  def test_aten_addmm_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.addmm, args, kwargs)

  def test_aten_alias_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.alias, args, kwargs)

  def test_aten_alias_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.alias, args, kwargs)

  def test_aten_alias_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.alias, args, kwargs)

  def test_aten_amax_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.amax, args, kwargs)

  def test_aten_amax_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.amax, args, kwargs)

  def test_aten_amax_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.amax, args, kwargs)

  def test_aten_amin_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.amin, args, kwargs)

  def test_aten_amin_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.amin, args, kwargs)

  def test_aten_amin_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.amin, args, kwargs)

  def test_aten_any_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any, args, kwargs)

  def test_aten_any_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any, args, kwargs)

  def test_aten_any_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any, args, kwargs)

  def test_aten_any_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any.dim, args, kwargs)

  def test_aten_any_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any.dim, args, kwargs)

  def test_aten_any_dim_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any.dim, args, kwargs)

  def test_aten_any_dims_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any.dims, args, kwargs)

  def test_aten_any_dims_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any.dims, args, kwargs)

  def test_aten_any_dims_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.any.dims, args, kwargs)

  def test_aten_argmax_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.argmax, args, kwargs)

  def test_aten_argmax_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.argmax, args, kwargs)

  def test_aten_argmax_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.argmax, args, kwargs)

  def test_aten_argmin_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.argmin, args, kwargs)

  def test_aten_argmin_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.argmin, args, kwargs)

  def test_aten_argmin_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.argmin, args, kwargs)

  def test_aten_as_strided_copy_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [2, 2],
        [
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.as_strided_copy, args, kwargs)

  def test_aten_as_strided_copy_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [2, 2],
        [
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.as_strided_copy, args, kwargs)

  def test_aten_as_strided_copy_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [2, 2],
        [
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.as_strided_copy, args, kwargs)

  def test_aten_asin_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.asin, args, kwargs)

  def test_aten_asin_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.asin, args, kwargs)

  def test_aten_asin_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.asin, args, kwargs)

  def test_aten_asinh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.asinh, args, kwargs)

  def test_aten_asinh_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.asinh, args, kwargs)

  def test_aten_asinh_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.asinh, args, kwargs)

  def test_aten_atan_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atan, args, kwargs)

  def test_aten_atan_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atan, args, kwargs)

  def test_aten_atan_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atan, args, kwargs)

  def test_aten_atan2_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atan2, args, kwargs)

  def test_aten_atan2_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atan2, args, kwargs)

  def test_aten_atanh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atanh, args, kwargs)

  def test_aten_atanh_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atanh, args, kwargs)

  def test_aten_atanh_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.atanh, args, kwargs)

  def test_aten_avg_pool2d_0(self):
    args = (
        torch.randn((1, 3, 1, 10)).to(torch.float32),
        [
            1,
            2,
        ],
        [
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.avg_pool2d, args, kwargs)

  def test_aten_avg_pool2d_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.avg_pool2d, args, kwargs)

  def test_aten_avg_pool2d_2(self):
    args = (
        torch.randn((1, 192, 40, 40)).to(torch.float32),
        [3, 3],
        [1, 1],
        [1, 1],
        True,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.avg_pool2d, args, kwargs)

  def test_aten_avg_pool3d_0(self):
    args = (
        torch.randn((1, 3, 10, 10, 10)).to(torch.float32),
        [
            2,
            2,
            2,
        ],
        [
            2,
            2,
            2,
        ],
        [
            0,
            0,
            0,
        ],
        False,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.avg_pool3d, args, kwargs)

  def test_aten_bitwise_and_Scalar_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_and.Scalar, args,
                           kwargs)

  def test_aten_bitwise_and_Tensor_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_and.Tensor, args,
                           kwargs)

  def test_aten_bitwise_and_Tensor_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_and.Tensor, args,
                           kwargs)

  def test_aten_bitwise_and_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_and.Tensor, args,
                           kwargs)

  def test_aten_bitwise_and_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_and.Tensor, args,
                           kwargs)

  def test_aten_bitwise_or_Scalar_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_or.Scalar, args, kwargs)

  def test_aten_bitwise_xor_Scalar_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bitwise_xor.Scalar, args,
                           kwargs)

  def test_aten_bmm_0(self):
    args = (
        torch.randn((10, 10, 10)).to(torch.float32),
        torch.randn((10, 10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bmm, args, kwargs)

  def test_aten_bmm_1(self):
    args = (
        torch.randn((10, 10, 10)).to(torch.float16),
        torch.randn((10, 10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bmm, args, kwargs)

  def test_aten_bmm_2(self):
    args = (
        torch.randint(0, 10, (10, 10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.bmm, args, kwargs)

  def test_aten_cat_0(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cat, args, kwargs)

  def test_aten_cat_1(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cat, args, kwargs)

  def test_aten_cat_2(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cat, args, kwargs)

  def test_aten__cdist_forward_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        1.0,
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._cdist_forward, args, kwargs)

  def test_aten_ceil_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ceil, args, kwargs)

  def test_aten_ceil_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ceil, args, kwargs)

  def test_aten_ceil_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ceil, args, kwargs)

  def test_aten_clamp_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clamp, args, kwargs)

  def test_aten_clamp_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clamp, args, kwargs)

  def test_aten_clamp_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clamp, args, kwargs)

  def test_aten_clamp_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((1,)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clamp.Tensor, args, kwargs)

  def test_aten_clamp_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((1,)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clamp.Tensor, args, kwargs)

  def test_aten_clamp_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (1,)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clamp.Tensor, args, kwargs)

  def test_aten_clone_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clone, args, kwargs)

  def test_aten_clone_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clone, args, kwargs)

  def test_aten_clone_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.clone, args, kwargs)

  def test_aten_constant_pad_nd_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.constant_pad_nd, args, kwargs)

  def test_aten_constant_pad_nd_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.constant_pad_nd, args, kwargs)

  def test_aten_constant_pad_nd_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.constant_pad_nd, args, kwargs)

  def test_aten_convolution_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        torch.randn((2, 2, 2)).to(torch.float32),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.convolution, args, kwargs)

  def test_aten_convolution_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float16),
        torch.randn((2, 2, 2)).to(torch.float16),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.convolution, args, kwargs)

  def test_aten_convolution_2(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        torch.randint(0, 10, (2, 2, 2)).to(torch.int32),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.convolution, args, kwargs)

  def test_aten_cos_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cos, args, kwargs)

  def test_aten_cos_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cos, args, kwargs)

  def test_aten_cos_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cos, args, kwargs)

  def test_aten_cosh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cosh, args, kwargs)

  def test_aten_cosh_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cosh, args, kwargs)

  def test_aten_cosh_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cosh, args, kwargs)

  def test_aten_cumsum_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cumsum, args, kwargs)

  def test_aten_cumsum_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(
        self, torch.ops.aten.cumsum, args, kwargs, rtol=0.001, atol=0.01)

  def test_aten_cumsum_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.cumsum, args, kwargs)

  def test_aten_diagonal_0(self):
    args = (torch.randn((10, 20)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.diagonal, args, kwargs)

  def test_aten_diagonal_1(self):
    args = (torch.randn((10, 20)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.diagonal, args, kwargs)

  def test_aten_diagonal_2(self):
    args = (torch.randint(0, 10, (10, 20)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.diagonal, args, kwargs)

  def test_aten_div_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.5,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Scalar, args, kwargs)

  def test_aten_div_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.5,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Scalar, args, kwargs)

  def test_aten_div_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.5,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Scalar, args, kwargs)

  def test_aten_div_Scalar_mode_0(self):

    def aten_div_Scalar_mode_rounding_mode_trunc(input, other):
      return torch.ops.aten.div.Scalar_mode(input, other, rounding_mode='floor')

    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, aten_div_Scalar_mode_rounding_mode_trunc, args,
                           kwargs)

  def test_aten_div_Scalar_mode_1(self):

    def aten_div_Scalar_mode_rounding_mode_trunc(input, other):
      return torch.ops.aten.div.Scalar_mode(input, other, rounding_mode='floor')

    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, aten_div_Scalar_mode_rounding_mode_trunc, args,
                           kwargs)

  def test_aten_div_Scalar_mode_2(self):

    def aten_div_Scalar_mode_rounding_mode_trunc(input, other):
      return torch.ops.aten.div.Scalar_mode(input, other, rounding_mode='floor')

    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, aten_div_Scalar_mode_rounding_mode_trunc, args,
                           kwargs)

  def test_aten_div_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Tensor, args, kwargs)

  def test_aten_div_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Tensor, args, kwargs)

  def test_aten_div_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Tensor, args, kwargs)

  def test_aten_div_Tensor_3(self):
    args = (
        torch.rand(1, 3, 4, 1),
        torch.rand(10),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.div.Tensor, args, kwargs)

  def test_aten_div_Tensor_mode_0(self):

    def aten_div_Tensor_mode_rounding_mode_trunc(input, other):
      return torch.ops.aten.div.Tensor_mode(input, other, rounding_mode='trunc')

    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, aten_div_Tensor_mode_rounding_mode_trunc, args,
                           kwargs)

  def test_aten_div_Tensor_mode_1(self):

    def aten_div_Tensor_mode_rounding_mode_trunc(input, other):
      return torch.ops.aten.div.Tensor_mode(input, other, rounding_mode='trunc')

    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, aten_div_Tensor_mode_rounding_mode_trunc, args,
                           kwargs)

  def test_aten_embedding_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.embedding, args, kwargs)

  def test_aten_embedding_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.embedding, args, kwargs)

  def test_aten_embedding_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.embedding, args, kwargs)

  def test_aten_eq_Scalar_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.eq.Scalar, args, kwargs)

  def test_aten_eq_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.eq.Scalar, args, kwargs)

  def test_aten_eq_Scalar_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.eq.Scalar, args, kwargs)

  def test_aten_eq_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.eq.Tensor, args, kwargs)

  def test_aten_eq_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.eq.Tensor, args, kwargs)

  def test_aten_eq_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.eq.Tensor, args, kwargs)

  def test_aten_erf_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.erf, args, kwargs)

  def test_aten_erf_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.erf, args, kwargs)

  def test_aten_erf_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.erf, args, kwargs)

  def test_aten_exp_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.exp, args, kwargs)

  def test_aten_exp_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.exp, args, kwargs)

  def test_aten_exp_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.exp, args, kwargs)

  def test_aten_expand_0(self):
    args = (
        torch.randn((10, 1)).to(torch.float32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expand, args, kwargs)

  def test_aten_expand_1(self):
    args = (
        torch.randn((10, 1)).to(torch.float16),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expand, args, kwargs)

  def test_aten_expand_2(self):
    args = (
        torch.randint(0, 10, (10, 1)).to(torch.int32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expand, args, kwargs)

  def test_aten_expand_copy_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expand_copy, args, kwargs)

  def test_aten_expand_copy_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expand_copy, args, kwargs)

  def test_aten_expand_copy_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expand_copy, args, kwargs)

  def test_aten_expm1_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expm1, args, kwargs)

  def test_aten_expm1_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.expm1,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_expm1_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.expm1, args, kwargs)

  def test_aten_fill_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fill.Scalar, args, kwargs)

  def test_aten_fill_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fill.Scalar, args, kwargs)

  def test_aten_fill_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fill.Scalar, args, kwargs)

  def test_aten_fill_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn(()).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fill.Tensor, args, kwargs)

  def test_aten_fill_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn(()).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fill.Tensor, args, kwargs)

  def test_aten_fill_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, ()).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fill.Tensor, args, kwargs)

  def test_aten_flip_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.flip, args, kwargs)

  def test_aten_flip_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.flip, args, kwargs)

  def test_aten_flip_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.flip, args, kwargs)

  def test_aten_floor_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.floor, args, kwargs)

  def test_aten_floor_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.floor, args, kwargs)

  def test_aten_floor_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.floor, args, kwargs)

  def test_aten_floor_divide_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.floor_divide, args, kwargs)

  def test_aten_floor_divide_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.floor_divide, args, kwargs)

  def test_aten_fmod_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fmod.Scalar, args, kwargs)

  def test_aten_fmod_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fmod.Scalar, args, kwargs)

  def test_aten_fmod_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fmod.Scalar, args, kwargs)

  def test_aten_fmod_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fmod.Tensor, args, kwargs)

  def test_aten_fmod_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.fmod.Tensor, args, kwargs)

  def test_aten_full_like_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.full_like, args, kwargs)

  def test_aten_full_like_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.full_like, args, kwargs)

  def test_aten_full_like_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.full_like, args, kwargs)

  def test_aten_gather_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gather, args, kwargs)

  def test_aten_gather_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gather, args, kwargs)

  def test_aten_gather_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gather, args, kwargs)

  def test_aten_ge_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ge.Scalar, args, kwargs)

  def test_aten_ge_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ge.Scalar, args, kwargs)

  def test_aten_ge_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ge.Scalar, args, kwargs)

  def test_aten_ge_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ge.Tensor, args, kwargs)

  def test_aten_ge_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ge.Tensor, args, kwargs)

  def test_aten_ge_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ge.Tensor, args, kwargs)

  def test_aten_gelu_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gelu, args, kwargs)

  def test_aten_gelu_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.gelu,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_glu_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.glu, args, kwargs)

  def test_aten_glu_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.glu, args, kwargs)

  @unittest.skip
  def test_aten_grid_sampler_2d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 2, 2, 2)).to(torch.float32),
        0,
        0,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.grid_sampler_2d, args, kwargs)

  def test_aten_gt_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gt.Scalar, args, kwargs)

  def test_aten_gt_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gt.Scalar, args, kwargs)

  def test_aten_gt_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gt.Scalar, args, kwargs)

  def test_aten_gt_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gt.Tensor, args, kwargs)

  def test_aten_gt_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gt.Tensor, args, kwargs)

  def test_aten_gt_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.gt.Tensor, args, kwargs)

  def test_aten_hardtanh_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.hardtanh, args, kwargs)

  def test_aten_hardtanh_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.hardtanh, args, kwargs)

  def test_aten_hardtanh_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.hardtanh, args, kwargs)

  def test_aten_index_put_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randn((10,)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_put, args, kwargs)

  def test_aten_index_put_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randn((10,)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_put, args, kwargs)

  def test_aten_index_put_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randint(0, 10, (10,)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_put, args, kwargs)

  def test_aten_index_select_0(self):
    args = (
        torch.randn((2, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_select, args, kwargs)

  def test_aten_index_select_1(self):
    args = (
        torch.randn((2, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_select, args, kwargs)

  def test_aten_index_select_2(self):
    args = (
        torch.randint(0, 10, (2, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_select, args, kwargs)

  def test_aten_index_select_3(self):
    args = (
        torch.randn((2, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_select, args, kwargs)

  def test_aten_index_select_4(self):
    args = (
        torch.randn((2, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2,)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_select, args, kwargs)

  def test_aten_index_select_5(self):
    args = (
        torch.randint(0, 10, (2, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index_select, args, kwargs)

  def test_aten_index_Tensor_0(self):
    args = (
        torch.randn((11, 2)).to(torch.float32),
        [
            torch.randint(5, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index.Tensor, args, kwargs)

  def test_aten_index_Tensor_1(self):
    args = (
        torch.randn((11, 2)).to(torch.float16),
        [
            torch.randint(5, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index.Tensor, args, kwargs)

  def test_aten_index_Tensor_2(self):
    args = (
        torch.randint(0, 10, (11, 2)).to(torch.int32),
        [
            torch.randint(5, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.index.Tensor, args, kwargs)

  def test_aten_isinf_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.isinf, args, kwargs)

  def test_aten_isinf_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.isinf, args, kwargs)

  def test_aten_isinf_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.isinf, args, kwargs)

  def test_aten_isnan_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.isnan, args, kwargs)

  def test_aten_isnan_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.isnan, args, kwargs)

  def test_aten_isnan_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.isnan, args, kwargs)

  def test_aten_le_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.le.Scalar, args, kwargs)

  def test_aten_le_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.le.Scalar, args, kwargs)

  def test_aten_le_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.le.Scalar, args, kwargs)

  def test_aten_le_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.le.Tensor, args, kwargs)

  def test_aten_le_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.le.Tensor, args, kwargs)

  def test_aten_le_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.le.Tensor, args, kwargs)

  def test_aten_leaky_relu_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.leaky_relu, args, kwargs)

  def test_aten_leaky_relu_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.leaky_relu, args, kwargs)

  def test_aten_lift_fresh_copy_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lift_fresh_copy, args, kwargs)

  def test_aten_lift_fresh_copy_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lift_fresh_copy, args, kwargs)

  def test_aten_lift_fresh_copy_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lift_fresh_copy, args, kwargs)

  def test_aten_log_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(
        self, torch.ops.aten.log, args, kwargs, equal_nan=True)

  def test_aten_log_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(
        self, torch.ops.aten.log, args, kwargs, equal_nan=True)

  def test_aten_log_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log, args, kwargs)

  def test_aten_log10_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log10, args, kwargs)

  def test_aten_log10_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.log10,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_log10_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log10, args, kwargs)

  def test_aten_log1p_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log1p, args, kwargs)

  def test_aten_log1p_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log1p, args, kwargs)

  def test_aten_log1p_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log1p, args, kwargs)

  def test_aten_log2_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log2, args, kwargs)

  def test_aten_log2_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.log2,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_log2_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.log2, args, kwargs)

  def test_aten__log_softmax_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._log_softmax, args, kwargs)

  def test_aten__log_softmax_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        False,
    )
    kwargs = dict()
    run_export_and_compare(
        self, torch.ops.aten._log_softmax, args, kwargs, rtol=0.001, atol=0.01)

  def test_aten_logical_and_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_and, args, kwargs)

  def test_aten_logical_and_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_and, args, kwargs)

  def test_aten_logical_and_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_and, args, kwargs)

  def test_aten_logical_and_3(self):
    args = (
        torch.randint(0, 2, (10, 10)).to(torch.bool),
        torch.randint(0, 2, (10, 10)).to(torch.bool),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_and, args, kwargs)

  def test_aten_logical_not_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_not, args, kwargs)

  def test_aten_logical_not_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_not, args, kwargs)

  def test_aten_logical_not_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_not, args, kwargs)

  def test_aten_logical_or_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_or, args, kwargs)

  def test_aten_logical_or_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_or, args, kwargs)

  def test_aten_logical_or_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_or, args, kwargs)

  def test_aten_logical_xor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_xor, args, kwargs)

  def test_aten_logical_xor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_xor, args, kwargs)

  def test_aten_logical_xor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logical_xor, args, kwargs)

  def test_aten_logit_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logit, args, kwargs)

  def test_aten_logit_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logit, args, kwargs, rtol=1e-3)

  def test_aten_logit_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.logit, args, kwargs)

  def test_aten_lt_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lt.Scalar, args, kwargs)

  def test_aten_lt_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lt.Scalar, args, kwargs)

  def test_aten_lt_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lt.Scalar, args, kwargs)

  def test_aten_lt_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lt.Tensor, args, kwargs)

  def test_aten_lt_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lt.Tensor, args, kwargs)

  def test_aten_lt_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.lt.Tensor, args, kwargs)

  def test_aten_masked_fill_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.bool),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.masked_fill.Scalar, args,
                           kwargs)

  def test_aten_masked_fill_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.bool),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.masked_fill.Scalar, args,
                           kwargs)

  def test_aten_masked_fill_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randn((10, 10)).to(torch.bool),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.masked_fill.Scalar, args,
                           kwargs)

  def test_aten_max_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max.dim, args, kwargs)

  def test_aten_max_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max.dim, args, kwargs)

  def test_aten_max_dim_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max.dim, args, kwargs)

  def test_aten_max_pool2d_with_indices_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max_pool2d_with_indices, args,
                           kwargs)

  def test_aten_max_pool2d_with_indices_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float16),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max_pool2d_with_indices, args,
                           kwargs)

  def test_aten_max_pool2d_with_indices_2(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max_pool2d_with_indices, args,
                           kwargs)

  def test_aten_max_pool3d_with_indices_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max_pool3d_with_indices, args,
                           kwargs)

  def test_aten_max_pool3d_with_indices_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max_pool3d_with_indices, args,
                           kwargs)

  def test_aten_max_pool3d_with_indices_2(self):
    args = (
        torch.randint(0, 10, (1, 3, 2, 10)).to(torch.int32),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.max_pool3d_with_indices, args,
                           kwargs)

  def test_aten_maximum_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.maximum, args, kwargs)

  def test_aten_maximum_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.maximum, args, kwargs)

  def test_aten_maximum_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.maximum, args, kwargs)

  def test_aten_mean_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mean, args, kwargs)

  def test_aten_mean_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mean, args, kwargs)

  def test_aten_mean_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mean.dim, args, kwargs)

  def test_aten_mean_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mean.dim, args, kwargs)

  def test_aten_min_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.min.dim, args, kwargs)

  def test_aten_min_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.min.dim, args, kwargs)

  def test_aten_min_dim_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.min.dim, args, kwargs)

  def test_aten_minimum_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.minimum, args, kwargs)

  def test_aten_minimum_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.minimum, args, kwargs)

  def test_aten_minimum_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.minimum, args, kwargs)

  def test_aten_mm_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mm, args, kwargs)

  def test_aten_mm_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mm, args, kwargs)

  def test_aten_mm_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mm, args, kwargs)

  def test_aten_mul_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mul.Scalar, args, kwargs)

  def test_aten_mul_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mul.Scalar, args, kwargs)

  def test_aten_mul_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mul.Scalar, args, kwargs)

  def test_aten_mul_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mul.Tensor, args, kwargs)

  def test_aten_mul_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mul.Tensor, args, kwargs)

  def test_aten_mul_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.mul.Tensor, args, kwargs)

  def test_aten_native_dropout_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1.0,
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.native_dropout, args, kwargs)

  def test_aten_native_dropout_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1.0,
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.native_dropout, args, kwargs)

  def test_aten_native_group_norm_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        None,
        None,
        1,
        3,
        20,
        1,
        0.0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.native_group_norm, args, kwargs)

  def test_aten_native_group_norm_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        None,
        None,
        1,
        3,
        20,
        1,
        0.0,
    )
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.native_group_norm,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_native_layer_norm_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            1,
            3,
            2,
            10,
        ],
        None,
        None,
        0.0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.native_layer_norm, args, kwargs)

  def test_aten_ne_Scalar_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ne.Scalar, args, kwargs)

  def test_aten_ne_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ne.Scalar, args, kwargs)

  def test_aten_ne_Scalar_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ne.Scalar, args, kwargs)

  def test_aten_ne_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ne.Tensor, args, kwargs)

  def test_aten_ne_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ne.Tensor, args, kwargs)

  def test_aten_ne_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.ne.Tensor, args, kwargs)

  def test_aten_neg_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.neg, args, kwargs)

  def test_aten_neg_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.neg, args, kwargs)

  def test_aten_neg_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.neg, args, kwargs)

  def test_aten__pdist_forward_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1.0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._pdist_forward, args, kwargs)

  def test_aten_permute_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.permute, args, kwargs)

  def test_aten_permute_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.permute, args, kwargs)

  def test_aten_permute_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.permute, args, kwargs)

  def test_aten_permute_copy_0(self):
    args = (
        torch.randn((2, 2, 2)).to(torch.float32),
        [
            1,
            2,
            0,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.permute_copy, args, kwargs)

  def test_aten_permute_copy_1(self):
    args = (
        torch.randn((2, 2, 2)).to(torch.float16),
        [
            1,
            2,
            0,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.permute_copy, args, kwargs)

  def test_aten_permute_copy_2(self):
    args = (
        torch.randint(0, 10, (2, 2, 2)).to(torch.int32),
        [
            1,
            2,
            0,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.permute_copy, args, kwargs)

  @unittest.skip
  def test_aten_pixel_shuffle_0(self):
    args = (
        torch.randn((1, 3, 10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pixel_shuffle, args, kwargs)

  @unittest.skip
  def test_aten_pixel_shuffle_1(self):
    args = (
        torch.randn((1, 3, 10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pixel_shuffle, args, kwargs)

  @unittest.skip
  def test_aten_pixel_shuffle_2(self):
    args = (
        torch.randint(0, 10, (1, 3, 10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pixel_shuffle, args, kwargs)

  def test_aten_pow_Scalar_0(self):
    args = (
        1.123,
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Scalar, args, kwargs)

  def test_aten_pow_Tensor_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1.2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Tensor_Scalar, args, kwargs)

  def test_aten_pow_Tensor_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1.2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Tensor_Scalar, args, kwargs)

  def test_aten_pow_Tensor_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1.2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Tensor_Scalar, args, kwargs)

  def test_aten_pow_Scalar_1(self):
    args = (10000, torch.randn(16 * 8))
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Scalar, args, kwargs)

  def test_aten_pow_Tensor_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Tensor_Tensor, args, kwargs)

  def test_aten_pow_Tensor_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Tensor_Tensor, args, kwargs)

  def test_aten_pow_Tensor_Tensor_2(self):
    args = (
        torch.randint(0, 5, (10, 10)).to(torch.int32),
        torch.randint(0, 5, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.pow.Tensor_Tensor, args, kwargs)

  def test_aten_prod_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.prod, args, kwargs)

  def test_aten_prod_1(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.prod, args, kwargs)

  def test_aten_prod_dim_int_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.prod.dim_int, args, kwargs)

  def test_aten_prod_dim_int_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.prod.dim_int, args, kwargs)

  # Due to the way randperm isn't on device, we manually assert checks here instead of using
  # the existing test harness.
  def test_aten_randperm_0(self):
    args = (20,)
    kwargs = dict()
    pytorch = torch.randperm(20)

    xla = torch.randperm(20, device=xm.xla_device())
    xla_detached = xla.detach().cpu()

    # Check equal lengths and that the sorted sets are equal. Since these numbers are randomly
    # generated there's no way to check that pytorch == pytorch/xla.
    self.assertEqual(len(pytorch), len(xla))
    self.assertEqual(sorted(set(pytorch)), sorted(set(xla_detached)))

  def test_aten_reciprocal_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reciprocal, args, kwargs)

  def test_aten_reciprocal_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reciprocal, args, kwargs)

  def test_aten_reciprocal_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reciprocal, args, kwargs)

  def test_aten_reflection_pad1d_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad1d, args, kwargs)

  def test_aten_reflection_pad1d_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad1d, args, kwargs)

  def test_aten_reflection_pad2d_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad2d, args, kwargs)

  def test_aten_reflection_pad2d_1(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad2d, args, kwargs)

  def test_aten_reflection_pad3d_0(self):
    args = (
        torch.randn((3, 3, 3, 3, 3)).to(torch.float32),
        [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad3d, args, kwargs)

  def test_aten_reflection_pad3d_1(self):
    args = (
        torch.randn((3, 3, 3, 3, 3)).to(torch.float16),
        [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad3d, args, kwargs)

  def test_aten_reflection_pad3d_2(self):
    args = (
        torch.randint(0, 10, (3, 3, 3, 3, 3)).to(torch.int32),
        [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.reflection_pad3d, args, kwargs)

  def test_aten_relu_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.relu, args, kwargs)

  def test_aten_relu_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.relu, args, kwargs)

  def test_aten_relu_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.relu, args, kwargs)

  def test_aten_remainder_Scalar_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.remainder.Scalar, args, kwargs)

  def test_aten_remainder_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.remainder.Scalar, args, kwargs)

  def test_aten_remainder_Scalar_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.remainder.Scalar, args, kwargs)

  def test_aten_remainder_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.remainder.Tensor, args, kwargs)

  def test_aten_remainder_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.remainder.Tensor, args, kwargs)

  def test_aten_replication_pad2d_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.replication_pad2d, args, kwargs)

  def test_aten_replication_pad2d_1(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.replication_pad2d, args, kwargs)

  def test_aten_replication_pad3d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.replication_pad3d, args, kwargs)

  def test_aten_replication_pad3d_1(self):
    args = (
        torch.randint(0, 10, (1, 3, 2, 10)).to(torch.int32),
        [
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.replication_pad3d, args, kwargs)

  def test_aten_resize__0(self):
    args = (
        torch.randn((2, 5, 10)).to(torch.float32),
        [
            2,
            5,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.resize_, args, kwargs)

  def test_aten_resize__1(self):
    args = (
        torch.randn((2, 5, 10)).to(torch.float16),
        [
            2,
            5,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.resize_, args, kwargs)

  def test_aten_resize__2(self):
    args = (
        torch.randint(0, 10, (2, 5, 10)).to(torch.int32),
        [
            2,
            5,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.resize_, args, kwargs)

  def test_aten_roll_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.roll, args, kwargs)

  def test_aten_roll_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.roll, args, kwargs)

  def test_aten_roll_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.roll, args, kwargs)

  def test_aten_round_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.round, args, kwargs)

  def test_aten_round_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.round, args, kwargs)

  def test_aten_round_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.round, args, kwargs)

  def test_aten_rsqrt_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    # run_export_and_compare(self, torch.ops.aten.rsqrt, args, kwargs)
    run_export_and_compare(
        self,
        torch.ops.aten.rsqrt,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_rsqrt_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    # run_export_and_compare(self, torch.ops.aten.rsqrt, args, kwargs)
    run_export_and_compare(
        self,
        torch.ops.aten.rsqrt,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_rsqrt_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    # run_export_and_compare(self, torch.ops.aten.rsqrt, args, kwargs)
    run_export_and_compare(
        self,
        torch.ops.aten.rsqrt,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_rsub_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.rsub.Scalar, args, kwargs)

  def test_aten_rsub_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.rsub.Scalar, args, kwargs)

  def test_aten_rsub_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.rsub.Scalar, args, kwargs)

  def test_aten_scatter_add_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter_add, args, kwargs)

  def test_aten_scatter_add_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter_add, args, kwargs)

  def test_aten_scatter_add_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter_add, args, kwargs)

  def test_aten_scatter_reduce_two_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
        "sum",
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter_reduce.two, args,
                           kwargs)

  def test_aten_scatter_reduce_two_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float16),
        "sum",
    )
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.scatter_reduce.two,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_scatter_reduce_two_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        "sum",
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter_reduce.two, args,
                           kwargs)

  def test_aten_scatter_src_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter.src, args, kwargs)

  def test_aten_scatter_src_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter.src, args, kwargs)

  def test_aten_scatter_src_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter.src, args, kwargs)

  def test_aten_scatter_value_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter.value, args, kwargs)

  def test_aten_scatter_value_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter.value, args, kwargs)

  def test_aten_scatter_value_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.scatter.value, args, kwargs)

  def test_aten_select_copy_int_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select_copy.int, args, kwargs)

  def test_aten_select_copy_int_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select_copy.int, args, kwargs)

  def test_aten_select_copy_int_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select_copy.int, args, kwargs)

  def test_aten_select_int_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select.int, args, kwargs)

  def test_aten_select_int_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select.int, args, kwargs)

  def test_aten_select_int_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select.int, args, kwargs)

  def test_aten_select_scatter_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randint(0, 10, (10,)).to(torch.int64),
        1,
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select_scatter, args, kwargs)

  def test_aten_select_scatter_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randint(0, 10, (10,)).to(torch.int64),
        1,
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select_scatter, args, kwargs)

  def test_aten_select_scatter_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10,)).to(torch.int64),
        1,
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.select_scatter, args, kwargs)

  def test_aten_sigmoid_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sigmoid, args, kwargs)

  def test_aten_sigmoid_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sigmoid, args, kwargs)

  def test_aten_sigmoid_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sigmoid, args, kwargs)

  def test_aten_sign_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sign, args, kwargs)

  def test_aten_sign_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sign, args, kwargs)

  def test_aten_sign_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sign, args, kwargs)

  def test_aten_sin_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sin, args, kwargs)

  def test_aten_sin_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sin, args, kwargs)

  def test_aten_sin_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sin, args, kwargs)

  def test_aten_sinh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sinh, args, kwargs)

  def test_aten_sinh_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sinh, args, kwargs)

  def test_aten_sinh_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sinh, args, kwargs)

  def test_aten_slice_copy_Tensor_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice_copy.Tensor, args, kwargs)

  def test_aten_slice_copy_Tensor_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice_copy.Tensor, args, kwargs)

  def test_aten_slice_copy_Tensor_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice_copy.Tensor, args, kwargs)

  def test_aten_slice_scatter_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice_scatter, args, kwargs)

  def test_aten_slice_scatter_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice_scatter, args, kwargs)

  def test_aten_slice_scatter_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice_scatter, args, kwargs)

  def test_aten_slice_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice.Tensor, args, kwargs)

  def test_aten_slice_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice.Tensor, args, kwargs)

  def test_aten_slice_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.slice.Tensor, args, kwargs)

  def test_aten__softmax_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._softmax, args, kwargs)

  def test_aten__softmax_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten._softmax, args, kwargs)

  def test_aten_sort_0(self):
    args = (
        torch.reshape(torch.randperm(100), (10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sort, args, kwargs)

  def test_aten_sort_1(self):
    args = (
        torch.reshape(torch.randperm(100), (10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sort, args, kwargs)

  def test_aten_sort_2(self):
    args = (
        torch.reshape(torch.randperm(100), (10, 10)).to(torch.int32),
        True,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sort, args, kwargs)

  def test_aten_split_copy_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.split_copy.Tensor, args, kwargs)

  def test_aten_split_copy_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.split_copy.Tensor, args, kwargs)

  def test_aten_split_copy_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        2,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.split_copy.Tensor, args, kwargs)

  def test_aten_split_with_sizes_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            1,
            2,
            3,
            4,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.split_with_sizes, args, kwargs)

  def test_aten_split_with_sizes_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            1,
            2,
            3,
            4,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.split_with_sizes, args, kwargs)

  def test_aten_split_with_sizes_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            1,
            2,
            3,
            4,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.split_with_sizes, args, kwargs)

  def test_aten_sqrt_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sqrt, args, kwargs)

  def test_aten_sqrt_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sqrt, args, kwargs)

  def test_aten_sqrt_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sqrt, args, kwargs)

  def test_aten_squeeze_copy_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze_copy.dim, args, kwargs)

  def test_aten_squeeze_copy_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze_copy.dim, args, kwargs)

  def test_aten_squeeze_copy_dim_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze_copy.dim, args, kwargs)

  def test_aten_squeeze_dims_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dims, args, kwargs)

  def test_aten_squeeze_dims_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dims, args, kwargs)

  def test_aten_squeeze_dims_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.squeeze.dims, args, kwargs)

  def test_aten_stack_0(self):
    args = ([
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    ],)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.stack, args, kwargs)

  def test_aten_stack_1(self):
    args = ([
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    ],)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.stack, args, kwargs)

  def test_aten_stack_2(self):
    args = ([
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    ],)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.stack, args, kwargs)

  def test_aten_sub_Scalar_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sub.Scalar, args, kwargs)

  def test_aten_sub_Scalar_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sub.Scalar, args, kwargs)

  def test_aten_sub_Scalar_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sub.Scalar, args, kwargs)

  def test_aten_sub_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sub.Tensor, args, kwargs)

  def test_aten_sub_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sub.Tensor, args, kwargs)

  def test_aten_sub_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sub.Tensor, args, kwargs)

  def test_aten_sum_dim_IntList_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sum.dim_IntList, args, kwargs)

  def test_aten_sum_dim_IntList_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        None,
    )
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.sum.dim_IntList,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_sum_dim_IntList_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        None,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.sum.dim_IntList, args, kwargs)

  def test_aten_tan_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tan, args, kwargs)

  def test_aten_tan_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(
        self,
        torch.ops.aten.tan,
        args,
        kwargs,
        rtol=0.001,
        atol=0.01,
    )

  def test_aten_tan_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tan, args, kwargs)

  def test_aten_tanh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tanh, args, kwargs)

  def test_aten_tanh_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tanh, args, kwargs)

  def test_aten_tanh_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tanh, args, kwargs)

  def test_aten_topk_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.topk, args, kwargs)

  def test_aten_topk_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.topk, args, kwargs)

  def test_aten_topk_2(self):
    args = (
        torch.reshape(torch.randperm(100), (10, 10)).to(torch.int32),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.topk, args, kwargs)

  def test_aten_transpose_copy_int_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.transpose_copy.int, args,
                           kwargs)

  def test_aten_transpose_copy_int_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.transpose_copy.int, args,
                           kwargs)

  def test_aten_transpose_copy_int_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0,
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.transpose_copy.int, args,
                           kwargs)

  def test_aten_tril_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tril, args, kwargs)

  def test_aten_tril_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tril, args, kwargs)

  def test_aten_tril_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.tril, args, kwargs)

  def test_aten_trunc_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.trunc, args, kwargs)

  def test_aten_trunc_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.trunc, args, kwargs)

  def test_aten_trunc_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.trunc, args, kwargs)

  def test_aten_unbind_copy_int_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unbind_copy.int, args, kwargs)

  def test_aten_unbind_copy_int_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unbind_copy.int, args, kwargs)

  def test_aten_unbind_copy_int_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unbind_copy.int, args, kwargs)

  def test_aten_unsqueeze_copy_0(self):
    args = (
        torch.randn((2, 0, 2)).to(torch.float32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze_copy, args, kwargs)

  def test_aten_unsqueeze_copy_1(self):
    args = (
        torch.randn((2, 0, 2)).to(torch.float16),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze_copy, args, kwargs)

  def test_aten_unsqueeze_copy_2(self):
    args = (
        torch.randint(0, 10, (2, 0, 2)).to(torch.int32),
        1,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.unsqueeze_copy, args, kwargs)

  def test_aten_upsample_bilinear2d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            3,
            20,
        ],
        False,
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.upsample_bilinear2d, args,
                           kwargs)

  def test_aten_upsample_nearest2d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            3,
            20,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.upsample_nearest2d, args,
                           kwargs)

  def test_aten_var_correction_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.var.correction, args, kwargs)

  def test_aten_var_correction_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.var.correction, args, kwargs)

  def test_aten_var_correction_2(self):
    args = (torch.randn((10, 10)).to(torch.float32), 0)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.var.correction, args, kwargs)

  def test_aten_var_correction_3(self):
    args = (torch.randn((10, 10)).to(torch.float16), 0)
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.var.correction, args, kwargs)

  def test_aten_view_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.view, args, kwargs)

  def test_aten_view_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.view, args, kwargs)

  def test_aten_view_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.view, args, kwargs)

  def test_aten_view_copy_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            2,
            5,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.view_copy, args, kwargs)

  def test_aten_view_copy_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            2,
            5,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.view_copy, args, kwargs)

  def test_aten_view_copy_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            2,
            5,
            10,
        ],
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.view_copy, args, kwargs)

  def test_aten_where_self_0(self):
    args = (
        torch.randn((10, 10)).to(torch.bool),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.where.self, args, kwargs)

  def test_aten_where_self_1(self):
    args = (
        torch.randn((10, 10)).to(torch.bool),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float64),
    )
    kwargs = dict()
    run_export_and_compare(self, torch.ops.aten.where.self, args, kwargs)


if __name__ == '__main__':
  unittest.main()
