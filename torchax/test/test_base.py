import unittest
import torch
from torch.utils import _pytree as pytree

from torchax import tensor

TestCase = unittest.TestCase
main = unittest.main


def diff_output(testcase, output1, output2, rtol, atol, equal_nan=True):
  if isinstance(output1, torch.Tensor):
    testcase.assertIsInstance(output2, torch.Tensor)
    output2_cpu = output2.detach().cpu()
    if output2_cpu.dtype != output1.dtype:
      output2_cpu = output2_cpu.to(output1.dtype)
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


def run_function_and_compare(testcase,
                             func,
                             args,
                             kwargs,
                             atol=1e-3,
                             rtol=1e-5,
                             equal_nan=True,
                             ignore_indices=False):
  with testcase.subTest("torch_eval"):
    res = func(*args, **kwargs)
    with testcase.subTest("torchax_eval"):
      args2, kwargs2 = pytree.tree_map_only(torch.Tensor, tensor.move_to_device,
                                            (args, kwargs))
      res2 = func(*args2, **kwargs2)
      res2 = pytree.tree_map_only(tensor.Tensor, lambda t: t.torch(), res2)
      with testcase.subTest("torchax_diff:" + str(atol)):
        if ignore_indices and isinstance(res, tuple) and len(res) == 2:
          diff_output(
              testcase,
              res[0],
              res2[0],
              atol=atol,
              rtol=rtol,
              equal_nan=equal_nan)
        else:
          diff_output(
              testcase, res, res2, atol=atol, rtol=rtol, equal_nan=equal_nan)