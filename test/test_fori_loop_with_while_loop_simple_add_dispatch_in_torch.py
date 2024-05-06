import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop
from torch_xla.experimental.fori_loop import _xla_while_loop, get_module_parameters
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
      a = body_fun(a, b)
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

    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())

    def cond_fn(upper, lower, one_value, x, input_value, output_value):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value):
      new_lower = torch.add(one_value, lower)
      output_value = linear_0(input_value)
      weight = linear_0.weight  # not be used actually, initialized as placeholder xlacomputation requirement
      bias = linear_0.bias  # not be used actually, initialized as placeholder xlacomputation requirement
      return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
          one_value, x), input_value.clone(), bias.clone(), weight.clone(
          ), output_value.clone()

    upper = torch.tensor([1], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, device=xm.xla_device())
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    upper__, lower__, one_value__, torch_add_res__, input_value__, bias__, weight__, output_value_real__, = while_loop(
        cond_fn, body_fn,
        (upper, lower, one_value, init_val, l_in_0, output_value))

    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    return self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))

  def test_while_loop_tpu_simple_linear_clean(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())

    def cond_fn(upper, lower, one_value, x, input_value, output_value):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value):
      new_lower = torch.add(one_value, lower)
      output_value = linear_0(input_value)
      return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
        one_value, x), input_value.clone(), output_value.clone()
      # return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
      #     one_value, x), input_value.clone(), output_value.clone()

    upper = torch.tensor([1], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, device=xm.xla_device())
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    upper__, lower__, one_value__, torch_add_res__, input_value__, bias__, weight__, output_value_real__, = while_loop(
        cond_fn, body_fn,
        (upper, lower, one_value, init_val, l_in_0, output_value))

    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    return self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))

  def test_while_loop_tpu_simple_linear_class(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())

      def forward(self, upper, lower, one_value, x, input_value, output_value):

        def cond_fn(upper, lower, one_value, x, input_value, output_value):
          return lower[0] < upper[0]

        def body_fn(upper, lower, one_value, x, input_value, output_value):
          new_lower = torch.add(one_value, lower)
          output_value_real = self.linear(input_value)
          weight = self.linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
          bias = self.linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
          return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
              one_value, x), input_value.clone(
              ), output_value_real, weight.clone(), bias.clone()

        return while_loop(
            cond_fn, body_fn,
            (upper, lower, one_value, x, input_value, output_value))

    simple_with_linear = SimpleWithLinear()
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, device=xm.xla_device())
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    weight_0 = simple_with_linear.linear.weight
    bias_0 = simple_with_linear.linear.bias

    aaa = {
        "simple_with_linear":
            (simple_with_linear, (upper, lower, one_value, init_val, l_in_0,
                                  output_value))
    }

    upper__, lower__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
        upper, lower, one_value, init_val, l_in_0, output_value)

    # create same weight/bias liear model for compare
    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    linear_0.weight.data = weight__
    linear_0.bias.data = bias__
    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))
    return aaa

  def test_while_loop_tpu_simple_linear_class_clean(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # define simple linear model class
    class SimpleWithLinear(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())

      def forward(self, one_value, x, input_value):
        output_value_real = self.linear(input_value)
        torch_add_res = torch.add(one_value, x)
        return torch_add_res, output_value_real

    simple_with_linear = SimpleWithLinear()

    # define cond and body
    def cond_fn(upper, lower, one_value, x, input_value, output_value, *args):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value, *args):
      new_lower = torch.add(one_value, lower)
      output_value_real = simple_with_linear(one_value, x, input_value)
      res = [upper.clone(), new_lower.clone(), one_value.clone(), output_value_real[0], input_value.clone(), output_value_real[1]]
      res = get_module_parameters(res, simple_with_linear, contain_output_balue=True)
      return tuple(res)

    # simple_with_linear = SimpleWithLinear()
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([1], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, device=xm.xla_device())
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    aaa = {
        "simple_with_linear":
            (simple_with_linear, (upper, lower, one_value, init_val, l_in_0,
                                  output_value))
    }

    additional_inputs = get_module_parameters([], simple_with_linear, contain_output_balue=False)

    upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, output_value_real__, = _xla_while_loop(
      cond_fn, body_fn,
      (upper, lower, one_value, init_val, l_in_0, output_value), tuple(additional_inputs))

    print("while_loop run times: ", torch_add_res__)
    print("output_value_real__: ", output_value_real__)

    expected = simple_with_linear(one_value, init_val, l_in_0)
    expected = expected[-1]
    print("expected: ", expected)

    self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))
    return aaa

  def test_while_loop_tpu_simple_linear_class_clean_only_linear(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # define simple linear model class
    class SimpleWithLinear(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())

      def forward(self, x):
        x = self.linear(x)
        return x

    simple_with_linear = SimpleWithLinear()

    # define cond and body
    def cond_fn(upper, lower, one_value, x, input_value, output_value, *args):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value, *args):
      new_lower = torch.add(one_value, lower)
      output_value_real = simple_with_linear(input_value)
      res = [upper.clone(), new_lower.clone(), one_value.clone(), torch.add(one_value, x).clone(), input_value.clone(), output_value_real.clone()]
      res = get_module_parameters(res, simple_with_linear, contain_output_balue=False)
      return tuple(res)

    # simple_with_linear = SimpleWithLinear()
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([1], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, dtype=torch.float32, device=device) # xm.xla_device()) # float
    output_value = torch.zeros([20], dtype=torch.float32, device=device)
    print("output_value: ", output_value)

    aaa = {
        "simple_with_linear":
            (simple_with_linear, (upper, lower, one_value, init_val, l_in_0,
                                  output_value))
    }

    additional_inputs = get_module_parameters([], simple_with_linear, contain_output_balue=False)
    # print("additional_inputs: ", additional_inputs)

    # while_loop
    upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, output_value_real__, = while_loop(
      cond_fn, body_fn,
      (upper, lower, one_value, init_val, l_in_0, output_value))

    # _xla_while_loop
    # upper__, lower__, one_value__, torch_add_res__, input_value__, weight1__, bias1__, output_value_real__, = _xla_while_loop(
    #   cond_fn, body_fn,
    #   (upper, lower, one_value, init_val, l_in_0, output_value), tuple(additional_inputs))

    print("while_loop run times: ", torch_add_res__)
    print("output_value_real__: ", output_value_real__)

    expected = simple_with_linear(l_in_0)
    print("expected: ", expected)

    self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))
    return aaa


  def test_fori_loop_tpu_addition(self):

    xm.mark_step()
    device = xm.xla_device()

    lower = torch.tensor([2], dtype=torch.int32, device=device)
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)

    def body_fun(a, b):
      return torch.add(a, b)

    upper_, new_lower_, one_value_, add_res_x_, res_ = fori_loop(
        upper, lower, body_fun, one_value, init_val)
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

    upper_, lower_, one_value_, add_res_x_, l_in_i_plus_1_, weight_, bias_, l_out_ = fori_loop(
        upper, lower, linear_0, init_val, l_in_0)

    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    self.assertTrue(torch.all(torch.eq(expected, l_out_)))

  def test_get_xlacomputation(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(20, 5).to(xla_device)
    t2 = torch.randn(20, 5).to(xla_device)
    # expected_data_handles = torch_xla._XLAC._get_xla_computation([t1], [], False)
    print("finish test here")

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
