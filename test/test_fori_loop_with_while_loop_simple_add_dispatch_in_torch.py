import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
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

    def cond_fn(x): # x = (xi,)
      # ten = torch.ones(10, dtype=torch.int32, device=device)
      return x[0] <= x[1]

    def body_fn(x): # x = (xi,)
      # onei = torch.ones(1, dtype=torch.int32, device=device)
      return (x[0] +  x[0], x[1])

    device = xm.xla_device()
    xx = torch.ones(1, dtype=torch.int32, device=device)
    xy = torch.ones(10, dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (xx, xy))
    expected = _fake_while_loop(cond_fn, body_fn, x)
    self.assertEqual(expected, res)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
