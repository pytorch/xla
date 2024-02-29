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
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands


class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu(self):

    def cond_fn(x):
      return x.sum() <= 10

    def body_fn(x):
      return (x + 1,)

    device = xm.xla_device()
    x = torch.ones(1, dtype=torch.int, device=device)
    res = while_loop(cond_fn, body_fn, (x,))
    expected = _fake_while_loop(cond_fn, body_fn, x)
    self.assertEqual(expected, res)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
