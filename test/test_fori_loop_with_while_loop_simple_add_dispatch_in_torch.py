import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop, _xla_while_loop_get_xla_computation
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


def _fake_while_loop(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands

# def _fake_fori_loop(lower, upper, body_fun, *init_val):
#   (plus_value, init_val) = init_val
#   for i in range((upper - lower)[0]):
#     plus_value, init_val = body_fun(plus_value, init_val)
#   return init_val

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
  # --------------------------------------
  # while_loop + PyLoweringContext
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

  # while_loop + PyLoweringContext + linear
  def test_while_loop_tpu_simple_linear_outside_loop(self):

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

  def test_while_loop_tpu_simple_linear_class_inside_loop(self):

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

  # WIP for target while_loop + PyLoweringContext + linear
  def test_while_loop_tpu_simple_linear_class_inside_loop(self):

    xm.mark_step()
    device = xm.xla_device()
    #device = ''
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):

      def __init__(self):
        super().__init__()
        # self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("dec", torch.tensor(1))

      # def forward(self, iter, x):
      #     def cond_fn(it, x):
      #         return it - self.dec > 0

      #     def body_fn(it, x):
      #         return it - 1, self.linear(x)

      #     return while_loop(cond_fn, body_fn, (iter, x))

      def forward(self, iter, x):
          def cond_fn(it, x):
              return it - self.dec > 0

          def body_fn(it, x):
              return it - 1, self.linear(x)

          # return while_loop(cond_fn, body_fn, (iter, x))
          return _xla_while_loop_get_xla_computation(cond_fn, body_fn, (iter, x), ())

      # def forward(self, upper, lower, one_value, x, input_value, output_value):

      #   def cond_fn(upper, lower, one_value, x, input_value, output_value):
      #     return lower[0] < upper[0]

      #   def body_fn(upper, lower, one_value, x, input_value, output_value):
      #     new_lower = torch.add(one_value, lower)
      #     output_value_real = self.linear(input_value)
      #     weight = self.linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
      #     bias = self.linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
      #     return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
      #         one_value, x), input_value.clone(
      #         ), output_value_real, weight.clone(), bias.clone()

      #   return while_loop(
      #       cond_fn, body_fn,
      #       (upper, lower, one_value, x, input_value, output_value))

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    #breakpoint()
    input = torch.randn(2, 2).to(device)
    iter = torch.tensor(3, device=device)
    res = simple_with_linear(iter, input)

    return res

    # upper = torch.tensor([52], dtype=torch.int32, device=device)
    # lower = torch.tensor([0], dtype=torch.int32, device=device)
    # one_value = torch.tensor([1], dtype=torch.int32, device=device)
    # init_val = torch.tensor([1], dtype=torch.int32, device=device)
    # l_in_0 = torch.rand(10, device=xm.xla_device())
    # output_value = torch.zeros([20], dtype=torch.float32, device=device)

    # upper__, lower__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
    #     upper, lower, one_value, init_val, l_in_0, output_value)

    # # create same weight/bias liear model for compare
    # linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    # linear_0.weight.data = weight__
    # linear_0.bias.data = bias__
    # expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    # self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))
    # return aaa

  def test_while_loop_tpu_simple_linear_target_inside_loop(self):

    xm.mark_step()
    device = xm.xla_device()
    #device = ''
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = torch.nn.Linear(2, 2)
          self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x):
          def cond_fn(it, x):
              return it - self.dec > 0

          def body_fn(it, x):
              return it - 1, self.linear(x)

          return while_loop(cond_fn, body_fn, (iter, x))
      
    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    #breakpoint()
    input = torch.randn(2, 2).to(device)
    iter = torch.tensor(3, device=device)
    res = simple_with_linear(iter, input)

  def test_while_loop_tpu_MNIST_outside_loop(self):

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

  # WIP for while_loop + PyLoweringContext + MNIST & inside_loop
  def test_while_loop_tpu_MNIST_outside_loop(self):

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

  # ------------------------
  # _get_xla_computation
  # pass
  def test_while_loop_get_xlacomputation(self):

    xm.mark_step()
    device = xm.xla_device()
    t1 = torch.randn(20, 5).to(device)
    t2 = torch.randn(20, 5).to(device)
    t3 = torch.add(t1, t2)

    ### implement one new function for xlacomputation generation with post-order
    print("before run _get_xla_computation: !!!!!!!!!")
    res_xla_computation = torch_xla._XLAC._get_xla_computation([t3], [], True)
    print("after run _get_xla_computation: !!!!!!!!!")
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("print computation from _get_xla_computation: !!!!!!!!!")
      print(hlo_print)
    else:
      print("print computation from _get_xla_computation: null !!!!!!!!!!!!!")

  def test_while_loop_get_xlacomputation_directly(self):

    xm.mark_step()
    device = xm.xla_device()
    t1 = torch.randn(20, 5).to(device)
    t2 = torch.randn(20, 5).to(device)
    t3 = torch.add(t1, t2)

    ### implement one new function for xlacomputation generation with post-order
    print("before run _get_xla_computation: !!!!!!!!!")
    res_xla_computation = torch_xla._XLAC._get_xla_computation([t3], [], True)
    print("after run _get_xla_computation: !!!!!!!!!")
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("print computation from _get_xla_computation: !!!!!!!!!")
      print(hlo_print)
    else:
      print("print computation from _get_xla_computation: null !!!!!!!!!!!!!")

  def test_while_loop_get_xlacomputation_tpu_simple_linear_without_while_loop(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, x):
        x = self.linear(x)
        return x

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    input = torch.randn(2, 2).to(device)
    t3 = simple_with_linear(input)

    ### implement one new function for xlacomputation generation with post-order
    print("before run _get_xla_computation: !!!!!!!!!")
    res_xla_computation = torch_xla._XLAC._get_xla_computation([t3], [], True)
    print("after run _get_xla_computation: !!!!!!!!!")
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("print computation from _get_xla_computation: !!!!!!!!!")
      print(hlo_print)
    else:
      print("print computation from _get_xla_computation: null !!!!!!!!!!!!!")

  # _xla_while_loop_get_xla_computation + _get_xla_computation
  def test_while_loop_tpu_subtraction_get_xla_computation(self):

    device = xm.xla_device()

    def cond_fn(init, limit_value):
      return limit_value[0] <= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      two_value = limit_value.clone()
      return (torch.sub(init, one_value), two_value)

    init = torch.tensor([10], dtype=torch.int32, device=device)
    limit_value = torch.tensor([0], dtype=torch.int32, device=device)
    res = _xla_while_loop_get_xla_computation(cond_fn, body_fn, (init, limit_value), ())
    expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    self.assertEqual(expected, res)

  def test_while_loop_tpu_addition_get_xla_computation(self):

    device = xm.xla_device()

    def cond_fn(init, limit_value):
      return limit_value[0] >= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      return (torch.add(init, one_value), limit_value.clone())

    # TODO(@manfei): init and limit_value has to be torch.tensor.
    init = torch.tensor([0], dtype=torch.int32, device=device)
    limit_value = torch.tensor([10], dtype=torch.int32, device=device)
    res = _xla_while_loop_get_xla_computation(cond_fn, body_fn, (init, limit_value), ())
    expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    self.assertEqual(expected, res)

  def test_while_loop_tpu_subtraction_nested_get_xla_computation(self):

    device = xm.xla_device()

    def cond_fn(init, limit_value):
      return limit_value[0] <= init[0]

    def body_fn(init, limit_value):
      one_value = torch.ones(1, dtype=torch.int32, device=device)
      two_value = limit_value.clone()
      return (torch.sub(torch.sub(init, one_value), one_value), two_value)

    init = torch.tensor([10], dtype=torch.int32, device=device)
    limit_value = torch.tensor([0], dtype=torch.int32, device=device)
    res = _xla_while_loop_get_xla_computation(cond_fn, body_fn, (init, limit_value), ())
    expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    self.assertEqual(expected, res)

  # _xla_while_loop_get_xla_computation + _get_xla_computation + linear
  def test_while_loop_tpu_simple_linear_outside_loop_get_xla_computation(self):

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

    upper__, lower__, one_value__, torch_add_res__, input_value__, bias__, weight__, output_value_real__, = _xla_while_loop_get_xla_computation(
        cond_fn, body_fn,
        (upper, lower, one_value, init_val, l_in_0, output_value), ())

    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    return self.assertTrue(torch.all(torch.eq(expected, output_value_real__)))

  def test_while_loop_tpu_simple_linear_class_inside_loop_get_xla_computation(self):

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

        return _xla_while_loop_get_xla_computation(
            cond_fn, body_fn,
            (upper, lower, one_value, x, input_value, output_value), ())

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

  # while_loop + _get_xla_computation: WIP
  def test_while_loop_get_xlacomputation_tpu_simple_linear_while_loop(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, x):
        x = self.linear(x)
        return x

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    input = torch.randn(2, 2).to(device)
    t3 = simple_with_linear(input)

    def cond_fn(upper, lower, one_value, x, input_value, output_value, *args):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value, *args):
      new_lower = torch.add(one_value, lower)
      output_value = simple_with_linear(input_value)
      res = [upper.clone(), new_lower.clone(), one_value.clone(), torch.add(one_value, x), input_value.clone(), output_value.clone()]
      return tuple(res)

    ### implement one new function for xlacomputation generation with post-order
    print("before run _get_xla_computation: !!!!!!!!!")
    res_xla_computation = torch_xla._XLAC._get_xla_computation([t3], [], True)
    print("after run _get_xla_computation: !!!!!!!!!")
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("print computation from _get_xla_computation: !!!!!!!!!")
      print(hlo_print)
    else:
      print("print computation from _get_xla_computation: null !!!!!!!!!!!!!")

    ### get xlacomputation via PyLoweringContext
    body_ctx = torch_xla._XLAC.lowering.LoweringContext()
    body_ctx.set_name_string("bodyctx")
    body_ctx.buildforiloop(list(t3), [])
    body_hlo = body_ctx.hlo()
    body_computation = xb.computation_from_module_proto("bodycomputation",
                                                        body_hlo)
    body_hlo_print = xb.get_computation_hlo(body_computation)
    print("print computation from PyLoweringContext: !!!!!!!!!")
    print(body_hlo_print)

  # fori_loop + PyLoweringContext: WIP
  def test_fori_loop_tpu_addition(self):

    xm.mark_step()
    device = xm.xla_device()

    lower = torch.tensor([2], dtype=torch.int32, device=device)
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    plus_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)

    def body_fun(*argus):
      plus_value, init_val = argus
      return plus_value.clone(), torch.add(plus_value, init_val).clone()

    _, _, _, actual = fori_loop(upper, lower, body_fun, plus_value, init_val)
    expected = _fake_fori_loop(lower, upper, body_fun, plus_value, init_val)
    self.assertEqual(expected, actual)

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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
