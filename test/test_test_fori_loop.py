import os
import unittest

import torch
import torch_xla
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm

xm.mark_step()
device = xm.xla_device()
torch.set_grad_enabled(False)

def _fake_fori_loop(lower, upper, body_fun, init_val):
  for i in range((upper - lower)[0]):
    output_value = body_fun(init_val)
  return output_value


class WhileLoopSwitchTest(unittest.TestCase):

  def test_simple_linear_layer_model_original(self):
    xm.mark_step()

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

    upper__, lower__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
        upper, lower, one_value, init_val, l_in_0, output_value)

    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    linear_0.weight.data = weight__
    linear_0.bias.data = bias__
    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    print("torch_add_res__: ", torch_add_res__)
    print("res: ", output_value_real__)
    print("expected: ", expected)

  def test_simple_linear_layer_model_switch_lower_and_upper(self):
    xm.mark_step()
    class SimpleWithLinear2(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())

      def forward(self, lower, upper, one_value, x, input_value, output_value):

        def cond_fn(lower, upper, one_value, x, input_value, output_value):
          return lower[0] < upper[0]

        def body_fn(lower, upper, one_value, x, input_value, output_value):
          new_lower = torch.add(one_value, lower)
          output_value_real = self.linear(input_value)
          weight = self.linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
          bias = self.linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
          return new_lower.clone(), upper.clone(), one_value.clone(), torch.add(
              one_value, x), input_value.clone(
              ), output_value_real, weight.clone(), bias.clone()

        return while_loop(cond_fn, body_fn, (lower, upper, one_value, x, input_value, output_value))

    simple_with_linear = SimpleWithLinear2()

    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, device=xm.xla_device())
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    weight_0 = simple_with_linear.linear.weight
    bias_0 = simple_with_linear.linear.bias

    itemone__, itemtwo__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
        lower, upper, one_value, init_val, l_in_0, output_value)

    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    linear_0.weight.data = weight__
    linear_0.bias.data = bias__
    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    print("itemone__: ", itemone__)
    print("itemtwo__: ", itemtwo__)
    print("torch_add_res__: ", torch_add_res__)
    print("res: ", output_value_real__)
    print("expected: ", expected)

  def test_simple_linear_layer_model_switch_x_and_one_value(self):
    xm.mark_step()
    class SimpleWithLinear3(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())

      def forward(self, upper, lower, x, one_value, input_value, output_value):

        def cond_fn(upper, lower, x, one_value, input_value, output_value):
          return lower[0] < upper[0]

        def body_fn(upper, lower, x, one_value, input_value, output_value):
          new_lower = torch.add(one_value, lower)
          output_value_real = self.linear(input_value)
          weight = self.linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
          bias = self.linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
          return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
              one_value, x), input_value.clone(
              ), output_value_real, weight.clone(), bias.clone()

        return while_loop(
            cond_fn, body_fn,
            (upper, lower, x, one_value, input_value, output_value))

    simple_with_linear = SimpleWithLinear3()

    upper = torch.tensor([52], dtype=torch.int32, device=device)
    lower = torch.tensor([0], dtype=torch.int32, device=device)
    one_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)
    l_in_0 = torch.rand(10, device=xm.xla_device())
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    weight_0 = simple_with_linear.linear.weight
    bias_0 = simple_with_linear.linear.bias

    upper__, lower__, one_value__, torch_add_res__, input_value__, output_value_real__, weight__, bias__ = simple_with_linear(
        upper, lower, init_val, one_value, l_in_0, output_value)

    linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    linear_0.weight.data = weight__
    linear_0.bias.data = bias__
    expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)

    print("torch_add_res__: ", torch_add_res__)
    print("res: ", output_value_real__)
    print("expected: ", expected)

# ### ------------simple linear four------------(output_value, input_value)
# """
# # Failed Eroor
# F0508 17:15:30.765712  823587 debug_macros.h:20] Non-OK-status: status.status() status: 
# INVALID_ARGUMENT: The parameter of condition and body, the result of the body, and 
# init must all have the same shape; got Condition: (in: (s32[1], s32[1], s32[1], s32[1], f32[20], 
# /*index=5*/f32[20], f32[20,10], f32[10])) -> pred[]; body: (in: (s32[1], s32[1], s32[1], s32[1], 
# f32[10], /*index=5*/f32[20], f32[20,10], f32[10])) -> (s32[1], s32[1], s32[1], s32[1], f32[10], 
# /*index=5*/f32[20], f32[20,10], f32[20]); init: (s32[1], s32[1], s32[1], s32[1], f32[20], 
# /*index=5*/f32[20], f32[20,10], f32[10])..: 
# """
# class SimpleWithLinear(torch.nn.Module):

#   def __init__(self):
#     super().__init__()
#     self.linear = torch.nn.Linear(10, 20).to(xm.xla_device())

#   def forward(self, upper, lower, one_value, x, output_value, input_value):

#     def cond_fn(upper, lower, one_value, x, output_value, input_value):
#       return lower[0] < upper[0]

#     def body_fn(upper, lower, one_value, x, output_value, input_value):
#       new_lower = torch.add(one_value, lower)
#       output_value_real = self.linear(input_value)
#       weight = self.linear.weight  # not be used actually, initialized as placeholder xlacomputation requirement
#       bias = self.linear.bias  # not be used actually, initialized as placeholder xlacomputation requirement
#       return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
#           one_value, x), input_value.clone(
#           ), output_value_real, weight.clone(), bias.clone()

#     return while_loop(
#         cond_fn, body_fn,
#         (upper, lower, one_value, x, output_value, input_value))

# simple_with_linear = SimpleWithLinear()
# upper = torch.tensor([52], dtype=torch.int32, device=device)
# lower = torch.tensor([0], dtype=torch.int32, device=device)
# one_value = torch.tensor([1], dtype=torch.int32, device=device)
# init_val = torch.tensor([1], dtype=torch.int32, device=device)
# l_in_0 = torch.rand(10, device=xm.xla_device())
# output_value = torch.zeros([20], dtype=torch.float32, device=device)

# weight_0 = simple_with_linear.linear.weight
# bias_0 = simple_with_linear.linear.bias

# upper__, lower__, one_value__, torch_add_res__, output_value_real__, weight__, bias__ = simple_with_linear(
#     upper, lower, one_value, init_val, output_value, l_in_0)

# linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
# linear_0.weight.data = weight__
# linear_0.bias.data = bias__
# expected = _fake_while_loop(lower, upper, linear_0, l_in_0)

# print("torch_add_res__: ", torch_add_res__)
# print("res: ", output_value_real__)
# print("expected: ", expected)

def test_simple_add_original(self):
  xm.mark_step()
  def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
      operands = body_fn(*operands)
    return operands

  def cond_fn(init, limit_value):
    return limit_value[0] <= init[0]

  def body_fn(init, limit_value):
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    return (torch.add(init, one_value), limit_value.clone())

  init = torch.tensor([0], dtype=torch.int32, device=device)
  limit_value = torch.tensor([10], dtype=torch.int32, device=device)
  res = while_loop(cond_fn, body_fn, (init, limit_value))
  expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))

  print("res: ", res)
  print("expected: ", expected)
  print("finish test")

def test_simple_add_return_output_value(self):
  xm.mark_step()
  def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
      operands = body_fn(*operands)
    return operands

  def cond_fn(init, limit_value, output_value):
    return limit_value[0] <= init[0]

  def body_fn(init, limit_value, output_value):
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    return (torch.add(init, one_value), limit_value.clone(), output_value.clone())

  init = torch.tensor([0], dtype=torch.int32, device=device)
  limit_value = torch.tensor([10], dtype=torch.int32, device=device)
  output_value = torch.tensor([10], dtype=torch.int32, device=device)
  res = while_loop(cond_fn, body_fn, (init, limit_value, output_value))
  expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value, output_value))

  # print("finish test: ", torch.eq(expected, res))
  print("res: ", res)
  print("expected: ", expected)
  print("finish test")

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
