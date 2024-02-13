import os
import unittest
from typing import Callable, Dict, List

import torch
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb

_TORCH_WHILE_LOOP_OPS = [
    torch._higher_order_ops.while_loop,
]

def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands

class WhileLoopTest(unittest.TestCase):

    def test_while_loop_tpu(self):
        def cond_fn(x):
            # z = torch.tensor(10, device=xm.xla_device())
            # return x < z
            return x < xb.Op.scalar(x.builder(), 10, dtype=xb.Type.S32)

        def body_fn(x):
            # y = torch.tensor(1, device=xm.xla_device())
            # return (x + y,)
            x = x + xb.Op.scalar(x.builder(), 1, dtype=xb.Type.S32)
            return xb.Op.tuple((x,))

        device = xm.xla_device()
        x = torch.ones(1, device=device)
        res = while_loop(cond_fn, body_fn, (x, ))
        expected = _fake_while_loop(cond_fn, body_fn, (x, ))
        self.assertEqual(expected, res)



if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)