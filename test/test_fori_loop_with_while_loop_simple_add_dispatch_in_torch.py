import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop, _xla_while_loop_target, _xla_while_loop_target_first, insert_model_pars_into_additional_inputs
# from torch_xla.experimental.fori_loop import _post_order_get_xla_computation_target_first, _xla_while_loop_get_xla_computation
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
import torch_xla.utils.utils as xu

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
  # passed
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

  # passed
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

  # passed
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
  # passed
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

  # passed
  def test_while_loop_tpu_simple_linear_class_inside_loop_while_loop(self):

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

  # passed
  def test_while_loop_tpu_simple_linear_target_inside_loop_while_loop(self):
    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

      def forward(self, iter, x):
        weight_bias_lists = []
        for name, param in self.linear.named_parameters():
          print("name: ", name)
          print("param: ", param)
          weight_bias_lists.append(param)

        def cond_fn(it, x):
          return it > 0

        def body_fn(it, x):
          return it - 1, self.linear(x)

        # torch.while_loop's additional_inputs could not be used in xlacomputation generation, looks like not real args
        return _xla_while_loop_target_first(cond_fn, body_fn, (iter, x), weight_bias_lists)

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    #breakpoint()
    input = torch.randn(2, 2).to(device)
    iter = torch.tensor(3, device=device)
    res = simple_with_linear(iter, input)
    print("act-res: ", res[-1])

    # create same weight/bias liear model for compare
    linear_0 = torch.nn.Linear(2, 2).to(device)
    linear_0.weight.data = simple_with_linear.linear.weight
    linear_0.bias.data = simple_with_linear.linear.bias
    # expected = _fake_fori_loop(lower, upper, linear_0, l_in_0)
    for i in range(3):
      input = linear_0(input)
    print("expexted: ", input)

    self.assertTrue(torch.all(res[-1].eq(input)))

  # WIP for target while_loop + PyLoweringContext + linear
  @unittest.skip("skip _get_xlacomputation now")
  def test_while_loop_tpu_simple_linear_class_inside_loop_xla_while_loop_get_xla_computation(self):

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
          # return _xla_while_loop_get_xla_computation(cond_fn, body_fn, (iter, x), ())

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    #breakpoint()
    input = torch.randn(2, 2).to(device)
    iter = torch.tensor(3, device=device)
    res = simple_with_linear(iter, input)

    return res

  class MNIST(torch.nn.Module):
    def __init__(self):
      # super().__init__()
      super(MNIST, self).__init__()
      self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2) # .to(xm.xla_device())
      self.bn1 = torch.nn.BatchNorm2d(10) # .to(xm.xla_device())
      self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5) # .to(xm.xla_device())
      self.bn2 = torch.nn.BatchNorm2d(20) # .to(xm.xla_device())
      self.fc1 = torch.nn.Linear(500, 50) # .to(xm.xla_device())
      self.fc2 = torch.nn.Linear(50, 10) # .to(xm.xla_device())

    def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = self.bn1(x)
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = self.bn2(x)
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

  @unittest.skip("skip _get_xlacomputation now")
  def test_while_loop_tpu_MNIST_outside_loop(self):

    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # linear_0 = torch.nn.Linear(10, 20).to(xm.xla_device())
    mnist = MNIST().to(device)

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

  # @unittest.skip("skip _get_xlacomputation now")
  def test_while_loop_tpu_MNIST_target_inside_loop_while_loop(self):
    xm.mark_step()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    n_epochs = 3
    batch_size_train = 8 # 64
    batch_size_test = 10 # 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    ### load data
    test_loader = xu.SampleGenerator(
    data=(torch.zeros(8, 1, 28,28), torch.zeros(8, dtype=torch.int64)),
    sample_count=1000 // 8 // xm.xrt_world_size())

    class MNIST(torch.nn.Module):
      def __init__(self):
        super().__init__()
        # self.linear = torch.nn.Linear(2, 2)
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2) # .to(xm.xla_device())
        self.bn1 = torch.nn.BatchNorm2d(10) # .to(xm.xla_device())
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5) # .to(xm.xla_device())
        self.bn2 = torch.nn.BatchNorm2d(20) # .to(xm.xla_device())
        self.fc1 = torch.nn.Linear(500, 50) # .to(xm.xla_device())
        self.fc2 = torch.nn.Linear(50, 10) # .to(xm.xla_device())
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []

      def forward(self, iter, x, y):
        # weight_bias_lists0 = []
        # bn_weight_bias_lists0 = []

        def cond_fn(it, x, y):
          return it > 1

        def body_fn(it, x, y):
          x = F.relu(F.max_pool2d(self.conv1(x), 2))
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          x = self.bn1(x)
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          # insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())
          x = F.relu(F.max_pool2d(self.conv2(x), 2))
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          x = self.bn2(x)
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          # insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          x = torch.flatten(x, 1)
          x = F.relu(self.fc1(x))
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          x = self.fc2(x)
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          # *
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          # self.bn_weight_bias_lists.reverse()
          # self.weight_bias_lists = self.weight_bias_lists + self.bn_weight_bias_lists
          # print("weight_bias_lists: ", weight_bias_lists)
          return it-1, x, F.log_softmax(x, dim=1)

        # bn_weight_bias_lists.reverse()
        # weight_bias_lists = weight_bias_lists + bn_weight_bias_lists
        # print("weight_bias_lists: ", weight_bias_lists)

        self.bn_weight_bias_lists.reverse()
        self.weight_bias_lists = self.weight_bias_lists + self.bn_weight_bias_lists
        return _xla_while_loop_target_first(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists)

        # return _xla_while_loop_target_first(cond_fn, body_fn, (iter, x, y), weight_bias_lists0)
        # return _xla_while_loop_target_first(cond_fn, body_fn, (iter, x), [])
        # return _post_order_get_xla_computation_target_first(cond_fn, body_fn, (iter, x), [])

        # def body_fn(it, x):
        #   x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.conv1.named_parameters())
        #   x = self.bn1(x)
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.bn1.named_parameters())
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.bn1.named_parameters())
        #   x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.conv2.named_parameters())
        #   x = self.bn2(x)
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.bn2.named_parameters())
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.bn2.named_parameters())
        #   x = torch.flatten(x, 1)
        #   x = F.relu(self.fc1(x))
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.fc1.named_parameters())
        #   x = self.fc2(x)
        #   # insert_model_pars_into_additional_inputs(weight_bias_lists, self.fc2.named_parameters())
        #   return it -1, F.log_softmax(x, dim=1)
        # return while_loop(cond_fn, body_fn, (iter, x))

    mnist = MNIST()
    mnist.to(device)
    #breakpoint()
    # input = torch.randn(2, 2).to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iter = torch.tensor(3, device=device)
    res = mnist(iter, l_in_0, l_out)
    print("res: ", res)
    # print("act-res: ", res[-1])

  @unittest.skip("skip _get_xlacomputation now")
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
  @unittest.skip("skip _get_xlacomputation now")
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
  @unittest.skip("skip _get_xlacomputation now")
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

  @unittest.skip("skip _get_xlacomputation now")
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

  @unittest.skip("skip _get_xlacomputation now")
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
  @unittest.skip("skip _get_xlacomputation now")
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

  @unittest.skip("skip _get_xlacomputation now")
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

  @unittest.skip("skip _get_xlacomputation now")
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
  @unittest.skip("skip _get_xlacomputation now")
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

  @unittest.skip("skip _get_xlacomputation now")
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
  @unittest.skip("skip _get_xlacomputation now")
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
  @unittest.skip("skip _get_xlacomputation now")
  def test_fori_loop_tpu_addition(self):

    xm.mark_step()
    device = xm.xla_device()

    lower = torch.tensor([2], dtype=torch.int32, device=device)
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    plus_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)

    def body_fun(*argus):
      plus_value, init_val = argus
      # return plus_value.clone(), torch.add(plus_value, init_val).clone()
      return plus_value, torch.add(plus_value, init_val)

    _, _, _, actual = fori_loop(upper, lower, body_fun, plus_value, init_val)
    expected = _fake_fori_loop(lower, upper, body_fun, plus_value, init_val)
    self.assertEqual(expected, actual)

  @unittest.skip("skip _get_xlacomputation now")
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
