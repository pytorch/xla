import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


def _fake_while_loop(cond_fn, body_fn, operands):
  while cond_fn(operands):
    operands = body_fn(operands)
  return operands


class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu(self):

    device = xm.xla_device()

    def cond_fn(x):
      ten = torch.ones(1, dtype=torch.int32, device=device)
      return x[0] >= ten[0]

    def body_fn(x):
      return (torch.sub(x[0], 1),)

    xi = torch.tensor([5], dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (xi,))
    expected = _fake_while_loop(cond_fn, body_fn, xi)
    self.assertEqual(expected, res)

  def test_while_loop_tpu_complex_situation(self):

    device = xm.xla_device()
    # run twice while_loop
    def cond_fn1(x):
      ten = torch.ones(1, dtype=torch.int32, device=device)
      return x[0] >= ten[0]

    def body_fn1(x):
      return (torch.sub(x[0], 1),)

    x1 = torch.tensor([10], dtype=torch.int32, device=device)
    res1 = while_loop(cond_fn1, body_fn1, (x1,))

    def cond_fn2(x):
      ten = torch.ones(1, dtype=torch.int32, device=device)
      return x[0] >= ten[0]

    def body_fn2(x):
      return (torch.sub(x[0], 1),)

    x2 = torch.tensor([5], dtype=torch.int32, device=device)
    res2 = while_loop(cond_fn2, body_fn2, (x2,))
    expected = _fake_while_loop(cond_fn2, body_fn2, x2)
    # print(met.metrics_report())
    self.assertEqual(expected, res2)

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
