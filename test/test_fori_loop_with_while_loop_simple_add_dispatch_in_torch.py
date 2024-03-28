import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


def _fake_while_loop(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands

def _fake_fori_loop(lower, upper, body_fun, *init_val):
  # operands need to be more than one here
  # print("upper - lower: ", upper - lower)
  # print("init_val: ", init_val)
  # print("type init_val: ", type(init_val))
  (a, b) = init_val
  # print("a: ", a)
  # print("b: ", b)
  for i in range((upper - lower)[0]):
    a = body_fun(a, b)
    # print("a: ", a)
    # print("i: ", i)
  return a

class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu_subtraction(self):

    device = xm.xla_device()

    def cond_fn(init, limit_value):
      return limit_value[0] <= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      two_value = limit_value.clone()
      return (torch.sub(init, one_value), two_value)

    init = torch.tensor([10], dtype=torch.int32, device=device)
    limit_value = torch.tensor([0], dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (init, limit_value))
    expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    self.assertEqual(expected, res)

  def test_while_loop_tpu_addition(self):

    device = xm.xla_device()

    def cond_fn(init, limit_value):
      return limit_value[0] >= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      return (torch.add(init, one_value), limit_value.clone())

    # TODO(@manfei): init and limit_value has to be torch.tensor.
    init = torch.tensor([0], dtype=torch.int32, device=device)
    limit_value = torch.tensor([10], dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (init, limit_value))
    expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    self.assertEqual(expected, res)

  def test_fori_loop_tpu_addition(self):

    xm.mark_step()
    device = xm.xla_device()

    # TODO(@manfei): lower, upper and init_val has to be torch.tensor.
    # init_val = torch.tensor([1], dtype=torch.int32, device=device)
    # one_value = torch.tensor([1], dtype=torch.int32, device=device)
    lower = torch.tensor([2], dtype=torch.int32, device=device)
    upper = torch.tensor([30], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    # one_value = torch.tensor([1], dtype=torch.int32, device=device)
    # init_val_list = (init_val, one_value)
    # lowers = torch.tensor(([1], [1], [1]), dtype=torch.int32, device=device) # lower, init_val, one_value

    def body_fun(a, b):
      return torch.add(a, b) # [0])
    # _, _, res, _ = fori_loop(lower, upper, body_fun, init_val, one_value) # init_val_list) # init_val)
    # A, B, res, D = fori_loop(lower, upper, body_fun, init_val, one_value) # init_val_list) # init_val)
    # A, B, res, D = fori_loop(upper, body_fun, lowers) # lower, upper, body_fun, init_val, one_value)
    # iter, upper, one_value, x
    # iter_, upper_, res, one_value_ = fori_loop(lower, upper, body_fun, init_val, one_value)
    # lower_, upper_, one_value_, res = fori_loop(lower, upper, body_fun, one_value, init_val)
    # upper_, lower_, one_value_, res = fori_loop(upper, lower, body_fun, one_value, init_val)
    # one_value, lower, upper, init_val
    # one_value_, lower_, upper_, res_ = fori_loop(upper, lower, body_fun, one_value, init_val)
    # upper, lower, one_value, init_val
    upper_, lower_, one_value_, res_ = fori_loop(upper, lower, body_fun, one_value, init_val)
    # upper, lower, init_val
    # upper_, lower_, res_ = fori_loop(upper, lower, body_fun, init_val)
    # print("one_value_: ", one_value_)
    print("lower_: ", lower_)
    print("upper_: ", upper_)
    print("one_value_: ", one_value_)
    print("res_: ", res_)
    # print("iter_: ", iter_)
    # print("upper_: ", upper_)
    # print("res: ", res)
    # print("one_value_: ", one_value_)
    # print("A: ", A) # lower_
    # print("B: ", B) # upper_
    # print("D: ", D) # one_value_
    # print("lower[0] <= upper[0]: ", lower[0] <= upper[0])
    # print("lower: ", lower)
    # print("upper: ", upper)
    # fori_loop(cond_fn, body_fn, (init, limit_value))
    expected = _fake_fori_loop(lower, upper, body_fun, init_val, one_value)
    print("expected: ", expected)
    self.assertEqual(expected, res_)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
