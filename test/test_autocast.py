import torch
import torch_xla
import torch_xla.core.xla_model as xm
import collections
import unittest
from torch.testing._internal.autocast_test_lists import AutocastTestLists
from torch_xla.amp import autocast, GradScaler

class TestAutocastBase(unittest.TestCase):

  def setUp(self):
    super(TestAutocastBase, self).setUp()
    self.autocast_lists = AutocastTestLists(torch.device("xla:0"))

  def tearDown(self):
    del self.autocast_lists
    super(TestAutocastBase, self).tearDown()

  def _run_autocast_outofplace(self,
                               op,
                               args,
                               run_as_type,
                               out_type=None,
                               module=torch,
                               add_kwargs=None):
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

    self.assertFalse(torch.is_autocast_enabled())
    with autocast():
      self.assertTrue(torch.is_autocast_enabled())

      out_type = out_type if out_type is not None else run_as_type
      output = output_method = None

      # Try module.* variant, if requested:
      if module is not None and hasattr(module, op):
        output = getattr(module, op)(*args, **add_kwargs)
        if isinstance(output, torch.Tensor):
          self.assertTrue(
              out_type == output.dtype,
              "autocast for torch.{} produced {}, should produce {}".format(
                  op, output.dtype, out_type))

      # Try Tensor.* variant:
      if hasattr(torch.Tensor, op):
        output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
        if isinstance(output_method, torch.Tensor):
          self.assertTrue(
              out_type == output_method.dtype,
              "autocast for torch.{} produced {}, should produce torch.{}"
              .format(op, output_method.dtype, out_type))

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
      with autocast(enabled=False):
        self.assertFalse(torch.is_autocast_enabled())

        if module is not None and hasattr(module, op):
          control = getattr(module, op)(*cast(args, run_as_type), **add_kwargs)
        else:
          control = getattr(args[0].to(run_as_type),
                            op)(*cast(args[1:], run_as_type), **add_kwargs)
        self.assertTrue(type(output_to_compare) == type(control))
        comparison = compare(output_to_compare, control)
        self.assertTrue(comparison,
                        "torch.{} result did not match control".format(op))
      self.assertTrue(torch.is_autocast_enabled())
    self.assertFalse(torch.is_autocast_enabled())


class TestAutocast(TestAutocastBase):

  def test_autocast_torch_fp16(self):
    with torch.backends.cudnn.flags(enabled=True, deterministic=True):
      for op_with_args in self.autocast_lists.torch_fp16:
        skip_test = False
        op, args = op_with_args[0], op_with_args[1]
        if len(op_with_args) == 3:
          skip_test = op_with_args[2]  # TEST_WITH_ROCM
        if not skip_test:
          self._run_autocast_outofplace(op, args, torch.float16)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
