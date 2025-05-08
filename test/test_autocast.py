import argparse
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import collections
import unittest
from torch.testing._internal.autocast_test_lists import AutocastTestLists
from torch_xla.amp import autocast, GradScaler


class AutocastTPUTestLists:
  # Supplies ops and arguments for TPU autocast tests
  def __init__(self, dev):
    super().__init__()
    n = 8
    # Utility arguments, created as one-element tuples
    pointwise0_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
    pointwise1_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
    pointwise2_bf16 = (torch.randn(n, dtype=torch.bfloat16, device=dev),)
    mat0_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
    mat1_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)
    mat2_bf16 = (torch.randn((n, n), dtype=torch.bfloat16, device=dev),)

    dummy_dimsets = ((n,), (n, n), (n, n, n), (n, n, n, n), (n, n, n, n, n))

    dummy_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),)
                  for dimset in dummy_dimsets]

    dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
    conv_args_bf16 = [(torch.randn(dimset, dtype=torch.bfloat16, device=dev),
                       torch.randn(dimset, dtype=torch.bfloat16, device=dev))
                      for dimset in dimsets]
    conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                       torch.randn(dimset, dtype=torch.float32, device=dev))
                      for dimset in dimsets]

    bias_fp32 = (torch.randn((n,), dtype=torch.float32, device=dev),)
    element0_fp32 = (torch.randn(1, dtype=torch.float32, device=dev),)
    pointwise0_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
    pointwise1_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)
    mat0_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
    mat1_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
    mat2_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
    mat3_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)

    dummy_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),)
                  for dimset in dummy_dimsets]
    # The lists below organize ops that autocast needs to test.
    # self.list_name corresponds to test_autocast_list_name .
    # Each op is associated with a tuple of valid arguments.

    # Some ops implement built-in type promotion.  These don't need autocasting,
    # but autocasting relies on their promotion, so we include tests to double-check.
    self.torch_expect_builtin_promote = [
        ("eq", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("ge", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("gt", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("le", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("lt", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("ne", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("add", pointwise0_fp32 + pointwise1_bf16, torch.float32),
        ("div", pointwise0_fp32 + pointwise1_bf16, torch.float32),
        ("mul", pointwise0_fp32 + pointwise1_bf16, torch.float32),
    ]
    self.methods_expect_builtin_promote = [
        ("__eq__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("__ge__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("__gt__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("__le__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("__lt__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("__ne__", pointwise0_fp32 + pointwise1_bf16, torch.bool),
        ("__add__", pointwise0_fp32 + pointwise1_bf16, torch.float32),
        ("__div__", pointwise0_fp32 + pointwise1_bf16, torch.float32),
        ("__mul__", pointwise0_fp32 + pointwise1_bf16, torch.float32),
    ]
    # The remaining lists organize ops that autocast treats explicitly.
    self.torch_bf16 = [
        ("conv1d", conv_args_fp32[0]),
        ("conv2d", conv_args_fp32[1]),
        ("conv3d", conv_args_fp32[2]),
        ("bmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                 torch.randn((n, n, n), device=dev, dtype=torch.float32))),
        ("mm", mat0_fp32 + mat1_fp32),
        ("matmul", mat0_fp32 + mat1_fp32),
        ("baddbmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32))),
        ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
        ("addbmm",
         mat0_fp32 + (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                      torch.randn((n, n, n), device=dev, dtype=torch.float32))),
        ("conv_tbc", (torch.randn((10, 7, 3), device=dev, dtype=torch.float32),
                      torch.randn((5, 3, 5), device=dev, dtype=torch.float32),
                      torch.randn(5, device=dev, dtype=torch.float32), 0)),
        ("conv_transpose1d", conv_args_fp32[0]),
        ("conv_transpose2d", conv_args_fp32[1]),
        ("conv_transpose3d", conv_args_fp32[2]),
        ("prelu", pointwise0_fp32 + element0_fp32),
        ("relu", pointwise0_fp32 + element0_fp32),
    ]
    self.torch_fp32 = [
        ("cosine_embedding_loss", (torch.tensor([[1, 2, 3]],
                                                device=dev,
                                                dtype=torch.bfloat16),
                                   torch.tensor([[1, 3, 4]],
                                                device=dev,
                                                dtype=torch.bfloat16),
                                   torch.tensor([1],
                                                device=dev,
                                                dtype=torch.int))),
        ("hinge_embedding_loss",
         mat0_bf16 + (torch.ones(n, device=dev, dtype=torch.int),)),
        ("margin_ranking_loss", mat0_bf16 + mat1_bf16 + (torch.ones(
            (n,), device=dev, dtype=torch.bfloat16),)),
        ("triplet_margin_loss", mat0_bf16 + mat1_bf16 + mat2_bf16),
        ("binary_cross_entropy_with_logits", mat0_bf16 + (torch.rand(
            (n, n), device=dev, dtype=torch.bfloat16),)),
    ]
    self.nn_bf16 = [
        ("linear", mat0_fp32 + mat1_fp32, {}),
    ]
    self.nn_fp32 = [
        ("nll_loss", (torch.rand((n, n), device=dev, dtype=torch.float),
                      torch.zeros((n,), device=dev, dtype=torch.long))),
        ("nll_loss2d", (torch.rand((n, n, n, n),
                                   device=dev,
                                   dtype=torch.bfloat16),
                        torch.zeros((n, n, n), device=dev, dtype=torch.long))),
        ("l1_loss", mat0_bf16 + mat1_bf16),
        ("smooth_l1_loss", mat0_bf16 + mat1_bf16),
        ("mse_loss", mat0_bf16 + mat1_bf16),
        ("multilabel_margin_loss", mat0_bf16 + (torch.ones(
            (n, n), device=dev, dtype=torch.long),)),
        ("soft_margin_loss", mat0_bf16 + (torch.ones(
            (n, n), device=dev, dtype=torch.long),)),
        ("multi_margin_loss", mat0_bf16 + (torch.ones(
            (n,), device=dev, dtype=torch.long),)),
    ]
    self.torch_need_autocast_promote = [
        ("cat", (pointwise0_bf16 + pointwise1_fp32,)),
        ("stack", (pointwise0_bf16 + pointwise1_fp32,)),
    ]
    self.methods_fp32 = []

    self.methods_bf16 = [("__matmul__", mat0_bf16 + mat1_fp32)]


class AutocastCudaTestExtraLists(object):

  def __init__(self, dev):
    super().__init__()
    n = 8
    dimsets = ((n, n, n), (n, n, n, n), (n, n, n, n, n))
    conv_args_fp32 = [(torch.randn(dimset, dtype=torch.float32, device=dev),
                       torch.randn(dimset, dtype=torch.float32, device=dev))
                      for dimset in dimsets]

    mat0_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
    mat1_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
    mat2_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)
    mat3_fp32 = (torch.randn((n, n), dtype=torch.float32, device=dev),)

    pointwise0_fp32 = (torch.randn(n, dtype=torch.float32, device=dev),)

    element0_fp32 = (torch.randn(1, dtype=torch.float32, device=dev),)

    # This is currently not part of AutocastTestLists and excludes `relu`, `addbmm`
    self.torch_bf16 = [
        ("conv1d", conv_args_fp32[0]),
        ("conv2d", conv_args_fp32[1]),
        ("conv3d", conv_args_fp32[2]),
        ("bmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                 torch.randn((n, n, n), device=dev, dtype=torch.float32))),
        ("mm", mat0_fp32 + mat1_fp32),
        ("matmul",
         torch.matmul(
             torch.ones([2, 3], device=dev, dtype=torch.float32),
             torch.ones([3, 2], device=dev, dtype=torch.float32))),
        ("baddbmm", (torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32),
                     torch.randn((n, n, n), device=dev, dtype=torch.float32))),
        ("addmm", mat1_fp32 + mat2_fp32 + mat3_fp32),
        ("conv_tbc", (torch.randn((10, 7, 3), device=dev, dtype=torch.float32),
                      torch.randn((5, 3, 5), device=dev, dtype=torch.float32),
                      torch.randn(5, device=dev, dtype=torch.float32), 0)),
        ("conv_transpose1d", conv_args_fp32[0]),
        ("conv_transpose2d", conv_args_fp32[1]),
        ("conv_transpose3d", conv_args_fp32[2]),
        ("prelu", pointwise0_fp32 + element0_fp32),
    ]


class AutocastCudaTestUnsupportedLists(object):

  def __init__(self):
    super().__init__()
    # Utility arguments, created as one-element tuples
    self.torch_expect_builtin_promote = [
        "cat",  # requires all input tensors to be the same type
        "equal",  # requires all input tensors to be the same type
        "stack",  # return f16 instead of f32
    ]
    self.methods_expect_builtin_promote = []

    # The remaining lists organize ops that autocast treats explicitly.
    self.torch_fp16 = [
        "_convolution_nogroup",  # need lowering
        "addmv",  # need lowering
    ]
    self.torch_fp32 = [
        "norm",  # produce f16 instead of f32
    ]
    self.torch_need_autocast_promote = [
        "scatter_add",  # cat currently requires all input tensors to be the same type
    ]
    self.nn_fp16 = []
    self.nn_fp32 = []
    self.linalg_fp16 = []
    self.methods_fp16 = []
    self.methods_fp32 = []
    self.banned = []


class TestAutocastBase(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.autocast_unsupported_lists = None

  @classmethod
  def tearDownClass(cls):
    del cls.autocast_lists

  def setUp(self):
    super(TestAutocastBase, self).setUp()
    self.is_autocast_enabled = None

  def tearDown(self):
    super(TestAutocastBase, self).tearDown()

  @classmethod
  def get_autocast_list(cls, list_name):
    if cls.autocast_unsupported_lists:
      return [
          tp for tp in getattr(cls.autocast_lists, list_name)
          if tp[0] not in getattr(cls.autocast_unsupported_lists, list_name)
      ]
    else:
      return [tp for tp in getattr(cls.autocast_lists, list_name)]

  def args_maybe_kwargs(self, op_with_args):
    if len(op_with_args) == 2:
      return op_with_args[0], op_with_args[1], {}
    else:
      return op_with_args[0], op_with_args[1], op_with_args[2]

  def _run_autocast_outofplace(self,
                               op,
                               args,
                               run_as_type,
                               out_type=None,
                               module=torch,
                               add_kwargs=None,
                               autocast_dtype=None):
    # helper to cast args
    def cast(val, to_type):
      if isinstance(val, torch.Tensor):
        return val.to(to_type) if val.is_floating_point() else val
      elif isinstance(val, collections.abc.Iterable):
        return type(val)(cast(v, to_type) for v in val)
      else:
        return val

    if add_kwargs is None:
      add_kwargs = {}

    self.assertFalse(self.is_autocast_enabled())
    with autocast(xm.xla_device(), dtype=autocast_dtype):
      self.assertTrue(self.is_autocast_enabled())

      out_type = out_type if out_type is not None else run_as_type
      output = output_method = None

      # Try module.* variant, if requested:
      if module is not None and hasattr(module, op):
        output = getattr(module, op)(*args, **add_kwargs)
        if isinstance(output, torch.Tensor):
          self.assertTrue(
              out_type == output.dtype,
              "autocast for {} produced {}, should produce {}".format(
                  op, output.dtype, out_type))

      # Try Tensor.* variant:
      if hasattr(torch.Tensor, op):
        output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
        if isinstance(output_method, torch.Tensor):
          self.assertTrue(
              out_type == output_method.dtype,
              "autocast for {} produced {}, should produce torch.{}".format(
                  op, output_method.dtype, out_type))

      self.assertTrue((output is not None) or (
          output_method is not None
      ), "{} not found as an attribute on either Tensor or the requested module {}"
                      .format(op, module))

      # Accounts for ops that return Tensors, iterables, and other non-Tensors.
      # For example, lstm_cell returns a tuple and equal returns bool.
      def compare(first, second):
        if isinstance(first, torch.Tensor):
          return torch.equal(first, second)
        elif isinstance(first, collections.abc.Iterable):
          return all(compare(f, s) for f, s in zip(first, second))
        else:
          return first == second

      # If both torch.* and Tensor.* variants were found, check outputs are identical
      if (output is not None) and (output_method is not None):
        self.assertTrue(type(output) == type(output_method))
        comparison = compare(output, output_method)
        self.assertTrue(
            comparison,
            "torch.{0} result did not match Tensor.{0} result".format(op))

      # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
      # as the C++-side autocasting, and should be bitwise accurate.
      output_to_compare = output if output is not None else output_method
      with autocast(xm.xla_device(), enabled=False):
        self.assertFalse(self.is_autocast_enabled())

        if module is not None and hasattr(module, op):
          control = getattr(module, op)(*cast(args, run_as_type), **add_kwargs)
        else:
          control = getattr(args[0].to(run_as_type),
                            op)(*cast(args[1:], run_as_type), **add_kwargs)
        self.assertTrue(type(output_to_compare) == type(control))
        comparison = compare(output_to_compare, control)
        self.assertTrue(comparison,
                        "torch.{} result did not match control".format(op))
      self.assertTrue(self.is_autocast_enabled())
    self.assertFalse(self.is_autocast_enabled())


@unittest.skipIf(not xm.get_xla_supported_devices("CUDA"),
                 f"CUDA autocast test.")
class TestAutocastCuda(TestAutocastBase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.autocast_lists = AutocastTestLists(torch.device(xm.xla_device()))
    cls.autocast_lists_extra = AutocastCudaTestExtraLists(
        torch.device(xm.xla_device()))
    cls.autocast_unsupported_lists = AutocastCudaTestUnsupportedLists()

  def setUp(self):
    super(TestAutocastCuda, self).setUp()
    self.is_autocast_enabled = torch.is_autocast_xla_enabled

  def test_autocast_nn_fp16(self):
    with torch.backends.cudnn.flags(enabled=True, deterministic=True):
      for op, args in TestAutocastCuda.get_autocast_list('nn_fp16'):
        self._run_autocast_outofplace(
            op, args, torch.float16, module=torch._C._nn)

  def test_autocast_linalg_fp16(self):
    with torch.backends.cudnn.flags(enabled=True, deterministic=True):
      for op, args in TestAutocastCuda.get_autocast_list('linalg_fp16'):
        self._run_autocast_outofplace(
            op, args, torch.float16, module=torch._C._linalg)

  def test_autocast_methods_fp16(self):
    with torch.backends.cudnn.flags(enabled=True, deterministic=True):
      for op, args in TestAutocastCuda.get_autocast_list('methods_fp16'):
        self._run_autocast_outofplace(op, args, torch.float16, module=None)

  def test_autocast_banned(self):
    with torch.cuda.amp.autocast():
      for op, args, module in TestAutocastCuda.get_autocast_list('banned'):
        with self.assertRaises(RuntimeError):
          getattr(module, op)(*args)

  def test_autocast_torch_fp32(self):
    for op_with_args in TestAutocastCuda.get_autocast_list('torch_fp32'):
      op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
      self._run_autocast_outofplace(
          op, args, torch.float32, add_kwargs=maybe_kwargs)

  def test_autocast_torch_bf16(self):
    bf16_test_list = [
        tp
        for tp in getattr(TestAutocastCuda.autocast_lists_extra, 'torch_bf16')
    ]
    for op_with_args in bf16_test_list:
      op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
      self._run_autocast_outofplace(
          op,
          args,
          torch.bfloat16,
          add_kwargs=maybe_kwargs,
          autocast_dtype=torch.bfloat16)

  def test_autocast_torch_need_autocast_promote(self):
    for op, args in TestAutocastCuda.get_autocast_list(
        'torch_need_autocast_promote'):
      self._run_autocast_outofplace(op, args, torch.float32)

  def test_autocast_torch_expect_builtin_promote(self):
    for op, args, out_type in TestAutocastCuda.get_autocast_list(
        'torch_expect_builtin_promote'):
      self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

  def test_autocast_nn_fp32(self):
    for op, args in TestAutocastCuda.get_autocast_list('nn_fp32'):
      self._run_autocast_outofplace(
          op, args, torch.float32, module=torch._C._nn)

  def test_autocast_methods_fp32(self):
    for op, args in TestAutocastCuda.get_autocast_list('methods_fp32'):
      self._run_autocast_outofplace(op, args, torch.float32, module=None)

  def test_autocast_methods_expect_builtin_promote(self):
    for op, args, out_type in TestAutocastCuda.get_autocast_list(
        'methods_expect_builtin_promote'):
      self._run_autocast_outofplace(
          op, args, torch.float32, module=None, out_type=out_type)


@unittest.skipIf(not xm.get_xla_supported_devices("TPU"), f"TPU autocast test.")
class TestAutocastTPU(TestAutocastBase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.autocast_lists = AutocastTPUTestLists(torch.device(xm.xla_device()))

  def setUp(self):
    super(TestAutocastTPU, self).setUp()
    self.is_autocast_enabled = torch.is_autocast_xla_enabled

  def test_autocast_methods_bf16(self):
    for op, args in TestAutocastTPU.get_autocast_list('methods_bf16'):
      self._run_autocast_outofplace(op, args, torch.bfloat16, module=None)

  def test_autocast_torch_fp32(self):
    for op_with_args in TestAutocastTPU.get_autocast_list('torch_fp32'):
      op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
      self._run_autocast_outofplace(
          op, args, torch.float32, add_kwargs=maybe_kwargs)

  def test_autocast_torch_need_autocast_promote(self):
    for op, args in TestAutocastTPU.get_autocast_list(
        'torch_need_autocast_promote'):
      self._run_autocast_outofplace(op, args, torch.float32)

  def test_autocast_torch_expect_builtin_promote(self):
    for op, args, out_type in TestAutocastTPU.get_autocast_list(
        'torch_expect_builtin_promote'):
      self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

  def test_autocast_nn_fp32(self):
    for op, args in TestAutocastTPU.get_autocast_list('nn_fp32'):
      self._run_autocast_outofplace(
          op, args, torch.float32, module=torch._C._nn)

  def test_autocast_methods_fp32(self):
    for op, args in TestAutocastTPU.get_autocast_list('methods_fp32'):
      self._run_autocast_outofplace(op, args, torch.float32, module=None)

  def test_autocast_methods_expect_builtin_promote(self):
    for op, args, out_type in TestAutocastTPU.get_autocast_list(
        'methods_expect_builtin_promote'):
      self._run_autocast_outofplace(
          op, args, torch.float32, module=None, out_type=out_type)

  def test_autocast_tpu_check_dtype(self):
    with autocast(xm.xla_device(), dtype=torch.float16):
      assert not torch.is_autocast_xla_enabled()


class TestOtherOps(unittest.TestCase):

  @unittest.skipIf(
      not xm.get_xla_supported_devices("GPU"),
      "the behavior of batch_norm autocast on GPU is different from others")
  def test_batch_norm_gpu(self):
    device = xm.xla_device()
    data = torch.randn(4, 16, 32, 32, device=device, dtype=torch.bfloat16)
    batch_norm = torch.nn.BatchNorm2d(16)
    with autocast(device, dtype=torch.bfloat16):
      output = batch_norm(data)
    torch_xla.sync()
    self.assertEqual(output.dtype, torch.bfloat16)

  # On TPU, the input of batch norm is casted into fp32, see torch_xla/csrc/autocast_mode.cpp
  @unittest.skipIf(
      not xm.get_xla_supported_devices("TPU"),
      "the behavior of batch_norm autocast on TPU is different from others")
  def test_batch_norm_tpu(self):
    device = xm.xla_device()
    data = torch.randn(4, 16, 32, 32, device=device, dtype=torch.bfloat16)
    batch_norm = torch.nn.BatchNorm2d(16)
    with autocast(device, dtype=torch.bfloat16):
      output = batch_norm(data)
    torch_xla.sync()
    self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  sys.exit(0 if test.result.wasSuccessful() else 1)
