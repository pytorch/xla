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
  if len(init_val) > 1:
    (a, b) = init_val
    for i in range((upper - lower)[0]):
      # a = body_fun(a, b)
      a = body_fun(*init_val)
  else:
    for i in range((upper - lower)[0]):
      a = body_fun(*init_val)
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

  def test_while_loop_tpu_subtraction_nested(self):

    device = xm.xla_device()

    def cond_fn(init, limit_value):
      return limit_value[0] <= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      two_value = limit_value.clone()
      return (torch.sub(torch.sub(init, one_value), one_value), two_value)

    init = torch.tensor([10], dtype=torch.int32, device=device)
    limit_value = torch.tensor([0], dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (init, limit_value))
    expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    self.assertEqual(expected, res)

  def test_while_loop_tpu_simple_linear(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device) # x
    l_in_0 = torch.randn(10, device=xm.xla_device()) # input_value
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    weight_0 = linear_0.weight
    bias_0 = linear_0.bias

    def cond_fn(upper, lower, one_value, x, input_value, weight_0, output_value, bias_0):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, weight_0, output_value, bias_0):
      new_lower = torch.add(one_value, lower)
      output_value = linear_0(input_value)
      weight = linear_0.weight
      bias = linear_0.bias
      return upper, new_lower, one_value, torch.add(one_value, x), input_value, weight, bias, output_value

    # print("!!! arrive here !!!")
    upper_, lower_, one_value_, add_res_x_, l_in_i_plus_1_, weight_, bias_, l_out_ = while_loop(cond_fn, body_fn, (upper, lower, one_value, init_val, l_in_0, weight_0, output_value, bias_0), additional_inputs=None)

    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    self.assertTrue(torch.all(torch.eq(expected, l_out_)))


  def test_fori_loop_tpu_addition(self):

    xm.mark_step()
    device = xm.xla_device()

    lower = torch.tensor([2], dtype=torch.int32, device=device)
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)

    def body_fun(a, b):
      return torch.add(a, b)

    lower_, upper_, res_ = fori_loop(upper, lower, body_fun, one_value,
                                     init_val)
    expected = _fake_fori_loop(lower, upper, body_fun, init_val, one_value)
    self.assertEqual(expected, res_)

  def test_fori_loop_tpu_simple_linear(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.randn(10, device=xm.xla_device())
    
    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())

    upper_, lower_, one_value_, add_res_x_, l_in_i_plus_1_, weight_, bias_, l_out_= fori_loop(upper, lower, linear_0, init_val, l_in_0)
    
    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    self.assertTrue(torch.all(torch.eq(expected, l_out_)))

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
