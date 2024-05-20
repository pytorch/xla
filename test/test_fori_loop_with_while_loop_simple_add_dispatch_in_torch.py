import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop, _xla_while_loop, _xla_while_loop_target, _xla_while_loop_target_first, insert_model_pars_into_additional_inputs, _xla_while_loop_target_first_second, _xla_while_loop_target_first_second_clean_version #,  _xla_while_loop_target_first_second_clean_version_s32
# from torch_xla.experimental.fori_loop import _post_order_get_xla_computation_target_first, _xla_while_loop_get_xla_computation
from torch_xla.experimental.fori_loop import _xla_while_loop_target_first_second_clean_version_s32_old, _xla_while_loop_target_first_second_clean_version_s32_may16_1530pm, _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm, _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
import torch_xla.utils.utils as xu

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def _fake_while_loop_second(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    # print("1 operands: ", operands)
    operands = body_fn(*operands)
    # print("operands: ", operands)
  return operands

def _fake_while_loop(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands

# def _fake_fori_loop(lower, upper, body_fun, *init_val):
  # (plus_value, init_val) = init_val
  # for i in range((upper - lower)[0]):
  #   plus_value, init_val = body_fun(plus_value, init_val)
  # return init_val

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

  # passed: torch pure version: subtraction
  def test_while_loop_tpu_subtraction_may16_2238pm_pure_torch(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x, y):
      return iteri > 0

    def body_fn(iteri, x, y):
      return iteri - 1, x, torch.sub(x, 1)

    init_val = torch.tensor(10)
    out_val = torch.tensor(15)
    # init_val = torch.tensor(10, dtype=torch.int32, device=device)
    # out_val = torch.tensor(15, dtype=torch.int32, device=device) # how to rand a torch.tensor?
    iteri = torch.tensor(3)
    # res =  _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    # res =  _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    res = while_loop(cond_fn, body_fn, (iteri, init_val, out_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val, out_val))
    print("expected: ", expected)
    # self.assertEqual(list(expected), res)

  # passed: torch_xla version: subtraction
  def test_while_loop_tpu_subtraction_may16_2238pm(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.sub(x, 1)

    # init_val = torch.tensor(10)
    # out_val = torch.tensor(15)
    init_val = torch.tensor(10, dtype=torch.int32, device=device)
    # out_val = torch.tensor(15, dtype=torch.int32, device=device) # how to rand a torch.tensor?
    # iteri = torch.tensor(3)
    iteri = torch.tensor(3, device=device)
    # res =  _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    # res =  _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    res = while_loop(cond_fn, body_fn, (iteri, init_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val))
    print("expected: ", expected)
    # self.assertEqual(list(expected), res)

  # passed: torch pure version: addition
  def test_while_loop_tpu_addition_may17_1149am_pure_torch(self):
    def cond_fn(iteri, x, y):
      return iteri > 0

    def body_fn(iteri, x, y):
      return iteri - 1, x, torch.add(x, 1)

    init_val = torch.tensor(3)
    out_val = torch.tensor(15)
    iteri = torch.tensor(10)
    res =  while_loop(cond_fn, body_fn, (iteri, init_val, out_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val, out_val))
    print("expected: ", expected)

  # passed: torch_xla version: addition
  def test_while_loop_tpu_addition_may17_1153am(self):
    xm.mark_step()
    device = xm.xla_device()
    # one_val = torch.tensor(1, dtype=torch.int32, device=device) # add this variable avoid body miss input arg in xla_computation, not fori_loop related

    def cond_fn(iteri, x): # make sure name of variable match
      return iteri > 0

    def body_fn(iteri, x):
      # return iter - one_val, x, torch.add(x, 2)
      # return iteri - 1, x, torch.add(x, 1) # 1 and 2 are different, so would not missed input
      return iteri - 1, torch.add(x, 1) # 1 and 2 are different, so would not missed input # due to `torch.while_loop's body_fn might be aliasing the input!`

    init_val = torch.tensor(3, dtype=torch.int32, device=device)
    # out_val = torch.tensor(15, dtype=torch.int32, device=device) # how to rand a torch.tensor?
    iteri = torch.tensor(10, device=device)
    res =  while_loop(cond_fn, body_fn, (iteri, init_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val))
    print("expected: ", expected)
    # self.assertEqual(list(expected), res)

  # def test_while_loop_tpu_addition_may16_2238pm(self):
  #   xm.mark_step()
  #   device = xm.xla_device()
  #   one_val = torch.tensor(1, dtype=torch.int32, device=device) # add this variable avoid body miss input arg in xla_computation, not fori_loop related

  #   def cond_fn(iteri, x, y): # make sure name of variable match
  #     return iteri > 0

  #   def body_fn(iteri, x, y):
  #     # return iter - one_val, x, torch.add(x, 2)
  #     return iteri - 1, x, torch.add(x, 1) # 1 and 2 are different, so would not missed input

  #   init_val = torch.tensor(3, dtype=torch.int32, device=device)
  #   out_val = torch.tensor(15, dtype=torch.int32, device=device) # how to rand a torch.tensor?
  #   # iteri = torch.tensor(10, dtype=torch.int32, device=device)
  #   iteri = torch.tensor(10, device=device)
  #   # res =  _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, init_val, out_val), [], [])
  #   res =  _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
  #   # iter_res, original_x, x_res
  #   # print("iter_res: ", iter_res)
  #   # print("original_x: ", original_x)
  #   # print("x_res: ", x_res)
  #   print("res: ", res)

  #   # expected = _fake_while_loop(cond_fn, body_fn, (iter, init_val, out_val))
  #   # print("expected: ", expected)
  #   # self.assertEqual(list(expected), res)

  # passed:  torch pure version: nestes addition
  def test_while_loop_tpu_addition_nested_may17_1456pm_pure_torch(self):

    def cond_fn(iteri, x, y):
      return iteri > 0

    def body_fn(iteri, x, y):
      return iteri - 1, x, torch.add(torch.add(x, 1), 1)

    init_val = torch.tensor(0)
    out_val = torch.tensor(0)
    iteri = torch.tensor(10)
    res =  while_loop(cond_fn, body_fn, (iteri, init_val, out_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val, out_val))
    print("expected: ", expected)
    self.assertEqual(expected, res)

  # limitsed passed: torch_xla version: nestes subtraction
  def test_while_loop_tpu_addition_nested_may17_1456pm(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(torch.add(x, 1), 1)

    # init_val = torch.tensor(0, dtype=torch.int32, device=device) # result would be wrong when init_val = 0, 1, due to body's xlacomputation missed inputs
    init_val = torch.tensor(2, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    res =  while_loop(cond_fn, body_fn, (iteri, init_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val))
    print("expected: ", expected)
    # self.assertEqual(expected, res)

  # def test_while_loop_tpu_subtraction_nested_may16_2238pm(self):
  #   xm.mark_step()
  #   device = xm.xla_device()

  #   def cond_fn(iter, x, y):
  #     return iter > 0

  #   def body_fn(iter, x, y):
  #     return iter - 1, x, torch.add(torch.add(x, 1), 1)

  #   init_val = torch.tensor(0, dtype=torch.int32, device=device)
  #   out_val = torch.tensor(0, dtype=torch.int32, device=device)
  #   iter = torch.tensor(10, device=device)
  #   res =  _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iter, init_val, out_val), [], [])
  #   print("res: ", res)
  #   # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
  #   # self.assertEqual(expected, res)

  # WIP: torch pure version: linear
  @unittest.skip("skip it now before debug")
  def test_while_loop_tpu_simple_linear_target_inside_loop_may17_1510pm_pure_torch(self):

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

    class SimpleLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        # self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x, y):

        def cond_fn(iter, x, y):
          # return iter > self.dec
          return iter > 0

        def body_fn(iter, x, y):
          y = self.linear(x)
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.linear.named_parameters())

          # return iter - self.dec, x, y #  why return x, y
          return iter - 1, x, y #  why return x, y

        # return _xla_while_loop_target_first_second_clean_version_s32(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32_old(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    linear_model = SimpleLinear()
    linear_model.to(device)
    #breakpoint()
    # input = torch.randn(2, 2).to(device)
    bs=16
    l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
    l_out = torch.randn(2, 2, dtype=torch.float32, device=device)
    iter = torch.tensor(3, device=device) # add dtype in iter would miss input
    res = linear_model(iter, l_in_0, l_out)
    print("res: ", res[-1])
    
    fake_linear = torch.nn.Linear(2, 2).to(xm.xla_device())
    fake_linear.weight.data = res[1]
    fake_linear.bias.data = res[2]
    # expected = _fake_while_loop_second(lower, upper, fake_linear, l_in_0)
    expected = fake_linear(l_in_0)
    print("expected: ", expected)

    print("l_in_0: ", l_in_0)

  # torch_xla version: linear
  def test_while_loop_tpu_simple_linear_target_inside_loop_may17_1515pm(self):
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

    class SimpleLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        # self.weight_bias_lists = []
        # self.bn_weight_bias_lists = []
        # self.register_buffer("dec", torch.tensor(1))

      def forward(self, iteri, x):

        def cond_fn(iteri, x):
          # return iter > self.dec
          return iteri > 0

        def body_fn(iteri, x):
          return iteri - 1, self.linear(x)

        return while_loop(cond_fn, body_fn, (iteri, x))

    linear_model = SimpleLinear()
    linear_model.to(device)
    bs=16
    l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
    # l_out = torch.randn(2, 2, dtype=torch.float32, device=device)
    iteri = torch.tensor(3, dtype=torch.int32, device=device) # add dtype in iter would miss input
    res = linear_model(iteri, l_in_0)
    print("res: ", res[-1])
    print("res are: ", res)

    # fake_linear = torch.nn.Linear(2, 2).to(xm.xla_device())
    # fake_linear.weight.data = res[1]
    # fake_linear.bias.data = res[2]
    # # expected = _fake_while_loop_second(lower, upper, fake_linear, l_in_0)
    # expected = fake_linear(l_in_0)
    # print("expected: ", expected)
    # print("l_in_0: ", l_in_0)

  def test_while_loop_tpu_MNIST_target_inside_loop_may16_2238pm(self):
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
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        # self.register_buffer("dec", torch.tensor(1))

      def forward(self, iteri, x, y):

        def cond_fn(iteri, x, y):
          return iteri > 0

        def body_fn(iteri, x, y):

          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          # add layers para manually
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())

          return iteri - 1, x, F.log_softmax(y, dim=1)

        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iteri, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32_old(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    mnist = MNIST()
    mnist.to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iteri = torch.tensor(3, dtype=torch.int64, device=device)
    # print("dtype iter: ", iter.dtype())
    # print("type iter: ", iter.type())
    # print("dtype ite 1: ", torch.dtype(iter))
    # print("dtype iter 2: ", iter.type().dtype)
    # print("dtype iter 3: ", iter.dtype)
    res = mnist(iteri, l_in_0, l_out)
    print("res: ", res[-1])

  # def test_while_loop_tpu_simple_linear_target_inside_loop_may16_2238pm(self):
  #   xm.mark_step()
  #   device = xm.xla_device()
  #   torch.set_grad_enabled(False)

  #   n_epochs = 3
  #   batch_size_train = 8 # 64
  #   batch_size_test = 10 # 1000
  #   learning_rate = 0.01
  #   momentum = 0.5
  #   log_interval = 10
  #   random_seed = 1
  #   torch.backends.cudnn.enabled = False
  #   torch.manual_seed(random_seed)

  #   ### load data
  #   test_loader = xu.SampleGenerator(
  #   data=(torch.zeros(8, 1, 28,28), torch.zeros(8, dtype=torch.int64)),
  #   sample_count=1000 // 8 // xm.xrt_world_size())

  #   class SimpleLinear(torch.nn.Module):
  #     def __init__(self):
  #       super().__init__()
  #       self.linear = torch.nn.Linear(2, 2)
  #       self.weight_bias_lists = []
  #       self.bn_weight_bias_lists = []
  #       # self.register_buffer("dec", torch.tensor(1))

  #     def forward(self, iter, x, y):

  #       def cond_fn(iter, x, y):
  #         # return iter > self.dec
  #         return iter > 0

  #       def body_fn(iter, x, y):
  #         y = self.linear(x)
  #         insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.linear.named_parameters())

  #         # return iter - self.dec, x, y #  why return x, y
  #         return iter - 1, x, y #  why return x, y

  #       # return _xla_while_loop_target_first_second_clean_version_s32(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
  #       # return _xla_while_loop_target_first_second_clean_version_s32_old(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
  #       return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

  #   linear_model = SimpleLinear()
  #   linear_model.to(device)
  #   #breakpoint()
  #   # input = torch.randn(2, 2).to(device)
  #   bs=16
  #   l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
  #   l_out = torch.randn(2, 2, dtype=torch.float32, device=device)
  #   iter = torch.tensor(3, device=device) # add dtype in iter would miss input
  #   res = linear_model(iter, l_in_0, l_out)
  #   print("res: ", res[-1])
    
  #   fake_linear = torch.nn.Linear(2, 2).to(xm.xla_device())
  #   fake_linear.weight.data = res[1]
  #   fake_linear.bias.data = res[2]
  #   # expected = _fake_while_loop_second(lower, upper, fake_linear, l_in_0)
  #   expected = fake_linear(l_in_0)
  #   print("expected: ", expected)

  #   print("l_in_0: ", l_in_0)

  # passed new nnn
  @unittest.skip("skip torch_xla func now")
  def test_while_loop_tpu_subtraction_s32(self):
    xm.mark_step()
    device = xm.xla_device()
    # one_value = torch.tensor(1, dtype=torch.int64, device=device)

    def cond_fn(iteri, x, y):
      # return iter > one_value
      return iteri > 0

    def body_fn(iteri, x, y):
      # return iter - one_value, x, torch.sub(x, one_value)
      return iteri - 1, x, torch.sub(x, 1)
      # return iter - 1, y, torch.sub(y, 1)

    init_val = torch.tensor(10, dtype=torch.int32, device=device)
    # out_val = torch.randint(0, 10, 1, dtype=torch.int32, device=device)
    out_val = torch.tensor(15, dtype=torch.int32, device=device) # how to rand a torch.tensor?
    # out_val = torch.rand(1, dtype=torch.int32, device=device)
    # iter = torch.tensor(10, dtype=torch.int32, device=device)
    iteri = torch.tensor(3, device=device)
    # res =  _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    res =  _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val, out_val))
    print("expected: ", expected)
    # self.assertEqual(list(expected), res)

  # passed new nnn unexpected
  @unittest.skip("skip torch_xla func now")
  def test_while_loop_tpu_addition_s32(self):
    xm.mark_step()
    device = xm.xla_device()
    one_val = torch.tensor(1, dtype=torch.int32, device=device) # add this variable avoid body miss input arg in xla_computation, not fori_loop related

    def cond_fn(iteri, x, y): # make sure name of variable match
      return iteri > 0

    def body_fn(iteri, x, y):
      # return iter - one_val, x, torch.add(x, 2)
      return iteri - 1, x, torch.add(x, 1) # 1 and 2 are different, so would not missed input

    init_val = torch.tensor(3, dtype=torch.int32, device=device)
    out_val = torch.tensor(15, dtype=torch.int32, device=device) # how to rand a torch.tensor?
    # iteri = torch.tensor(10, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    # res =  _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, init_val, out_val), [], [])
    res =  _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm(cond_fn, body_fn, (iteri, init_val, out_val), [], [])
    # iter_res, original_x, x_res
    # print("iter_res: ", iter_res)
    # print("original_x: ", original_x)
    # print("x_res: ", x_res)
    print("res: ", res)

    # expected = _fake_while_loop(cond_fn, body_fn, (iter, init_val, out_val))
    # print("expected: ", expected)
    # self.assertEqual(list(expected), res)

  @unittest.skip("skip _get_xlacomputation now")
  def test_while_loop_cal_get_xlacomputation(self):

    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iter, x, y): # make sure name of variable match
      return iter > 0

    def body_fn(iter, x, y):
      return iter - 1, x, torch.sub(x, 1) # torch.add(x, 1)

    init_val = torch.tensor(0, dtype=torch.int32, device=device)
    out_val = torch.tensor(15, dtype=torch.int32, device=device)
    iter = torch.tensor(10, device=device) # add dtype would miss input

    res = body_fn(iter, init_val, out_val)
    print("before run _get_xla_computation: !!!!!!!!!")
    res_xla_computation = torch_xla._XLAC._get_xla_computation(list(res), [], True)
    print("after run _get_xla_computation: !!!!!!!!!")
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("print computation from _get_xla_computation: !!!!!!!!!")
      print(hlo_print)
    else:
      print("print computation from _get_xla_computation: null !!!!!!!!!!!!!")

  # passed new nnn unexpected
  @unittest.skip("skip torch_xla func now")
  def test_while_loop_tpu_subtraction_nested_s32(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iter, x, y):
      return iter > 0

    def body_fn(iter, x, y):
      return iter - 1, x, torch.add(torch.add(x, 1), 1)

    init_val = torch.tensor(0, dtype=torch.int32, device=device)
    out_val = torch.tensor(0, dtype=torch.int32, device=device)
    iter = torch.tensor(10, device=device)
    res =  _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iter, init_val, out_val), [], [])
    print("res: ", res)
    # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    # self.assertEqual(expected, res)

  # passed now new ?
  @unittest.skip("skip torch_xla func now")
  def test_while_loop_tpu_MNIST_target_inside_loop_s32(self):
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
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        # self.register_buffer("dec", torch.tensor(1))

      def forward(self, iteri, x, y):

        def cond_fn(iteri, x, y):
          return iteri > 0

        def body_fn(iteri, x, y):

          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          # add layers para manually
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())

          return iteri - 1, x, F.log_softmax(y, dim=1)

        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iteri, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32_old(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    mnist = MNIST()
    mnist.to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iteri = torch.tensor(3, dtype=torch.int64, device=device)
    # print("dtype iter: ", iter.dtype())
    # print("type iter: ", iter.type())
    # print("dtype ite 1: ", torch.dtype(iter))
    # print("dtype iter 2: ", iter.type().dtype)
    # print("dtype iter 3: ", iter.dtype)
    res = mnist(iteri, l_in_0, l_out)
    print("res: ", res[-1])

  # unexpected
  @unittest.skip("skip torch_xla func now")
  def test_while_loop_tpu_simple_linear_target_inside_loop_s32(self):
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

    class SimpleLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        # self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x, y):

        def cond_fn(iter, x, y):
          # return iter > self.dec
          return iter > 0

        def body_fn(iter, x, y):
          y = self.linear(x)
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.linear.named_parameters())

          # return iter - self.dec, x, y #  why return x, y
          return iter - 1, x, y #  why return x, y

        # return _xla_while_loop_target_first_second_clean_version_s32(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32_old(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    linear_model = SimpleLinear()
    linear_model.to(device)
    #breakpoint()
    # input = torch.randn(2, 2).to(device)
    bs=16
    l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
    l_out = torch.randn(2, 2, dtype=torch.float32, device=device)
    iter = torch.tensor(3, device=device) # add dtype in iter would miss input
    res = linear_model(iter, l_in_0, l_out)
    print("res: ", res[-1])
    
    fake_linear = torch.nn.Linear(2, 2).to(xm.xla_device())
    fake_linear.weight.data = res[1]
    fake_linear.bias.data = res[2]
    # expected = _fake_while_loop_second(lower, upper, fake_linear, l_in_0)
    expected = fake_linear(l_in_0)
    print("expected: ", expected)

    print("l_in_0: ", l_in_0)



  # ==================================== other test ==========================================
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

  # --------------------------------------
  # while_loop + PyLoweringContext
  # passed
  @unittest.skip("skip old style now")
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
  @unittest.skip("skip old style now")
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
  @unittest.skip("skip old style now")
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
  @unittest.skip("skip old style now")
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
  @unittest.skip("skip old style now")
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
  @unittest.skip("skip old style now")
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

  @unittest.skip("skip now")
  # passed 
  def test_while_loop_tpu_MNIST_target_inside_loop_while_loop_clean_version_s64(self):
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
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        # self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x, y):

        def cond_fn(iter, x, y):
          return iter > 0

        def body_fn(iter, x, y):

          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          # add layers para manually
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())

          return iter - 1, x, F.log_softmax(y, dim=1)

        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return _xla_while_loop_target_first_second_clean_version_s32(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    mnist = MNIST()
    mnist.to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iter = torch.tensor(3, dtype=torch.int64, device=device)
    # print("dtype iter: ", iter.dtype())
    # print("type iter: ", iter.type())
    # print("dtype ite 1: ", torch.dtype(iter))
    # print("dtype iter 2: ", iter.type().dtype)
    # print("dtype iter 3: ", iter.dtype)
    res = mnist(iter, l_in_0, l_out)
    print("res: ", res[-1])

  # @unittest.skip("skip _get_xlacomputation now")
  # passed now
  @unittest.skip("skip s64 self.dec now")
  def test_while_loop_tpu_MNIST_target_inside_loop_while_loop_clean_version(self):
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
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x, y):

        def cond_fn(iter, x, y):
          return iter > self.dec

        def body_fn(iter, x, y):

          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          # add layers para manually
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())

          return iter - self.dec, x, F.log_softmax(y, dim=1)

        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    mnist = MNIST()
    mnist.to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iter = torch.tensor(3, device=device)
    res = mnist(iter, l_in_0, l_out)
    print("res: ", res[-1])

  # passed new
  @unittest.skip("skip old commented version now")
  def test_while_loop_tpu_MNIST_target_inside_loop_while_loop_comments_version(self):
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
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x, y):

        def cond_fn(iter, x, y):
          # return iter > 0
          return iter > self.dec

        def body_fn(iter, x, y): # def body_fn(iter, original_x, y):
          # x = original_x.clone()
          # x = original_x.clone()

          # x = F.relu(F.max_pool2d(self.conv1(x), 2))
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          # x = self.bn1(x)
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          # # insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())
          # x = F.relu(F.max_pool2d(self.conv2(x), 2))
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          # x = self.bn2(x)
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          # # insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          # x = torch.flatten(x, 1)
          # x = F.relu(self.fc1(x))
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          # x = self.fc2(x)
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          # insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.conv2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          # insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc1.named_parameters())
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.fc2.named_parameters())

          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())
          # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn1.named_parameters())
          # # insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.bn2.named_parameters())

          # self.bn_weight_bias_lists.append(self.dec)
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn2.named_parameters())
          insert_model_pars_into_additional_inputs(self.bn_weight_bias_lists, self.bn1.named_parameters())

          # keep this modification here due to the additional_inputs would be modified after body_xlacomputation triggered
          # self.bn_weight_bias_lists.reverse()
          # self.weight_bias_lists = self.weight_bias_lists + self.bn_weight_bias_lists
          # self.weight_bias_lists = [self.weight_bias_lists, self.bn_weight_bias_lists]
          # print("weight_bias_lists: ", weight_bias_lists)
          # for i in range(len(self.weight_bias_lists)): print("self.weight_bias_lists ", i, " size: ", self.weight_bias_lists[i].size())
          # return iter-1, x, F.log_softmax(x, dim=1)
          # return iter-1, original_x, F.log_softmax(x, dim=1)
          return iter - self.dec, x, F.log_softmax(y, dim=1)

        # self.bn_weight_bias_lists.reverse()
        # self.weight_bias_lists = self.weight_bias_lists + self.bn_weight_bias_lists
        # for i in range(len(self.weight_bias_lists)): print("now self.weight_bias_lists ", i, " size: ", self.weight_bias_lists[i].size())
        # return _xla_while_loop_target_first(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        return _xla_while_loop_target_first_second(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)
        # return while_loop(cond_fn, body_fn, (iter, x, y))
        # return _xla_while_loop(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists)

    mnist = MNIST()
    mnist.to(device)
    #breakpoint()
    # input = torch.randn(2, 2).to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iter = torch.tensor(3, device=device)
    res = mnist(iter, l_in_0, l_out)
    print("res: ", res[-1])
    # print("act-res: ", res[-1])
    # expected = _fake_fori_loop(0, 3, mnist, l_in_0)
    # for i in range(3):
    #   expected = mnist(l_in_0)
    # print("expected: ", expected)

  # passed new
  @unittest.skip("skip old version now")
  def test_while_loop_tpu_simple_linear_target_inside_loop_while_loop(self):
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

    class SimpleLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.weight_bias_lists = []
        self.bn_weight_bias_lists = []
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, iter, x, y):

        def cond_fn(iter, x, y):
          return iter > self.dec

        def body_fn(iter, x, y):
          y = self.linear(x)
          insert_model_pars_into_additional_inputs(self.weight_bias_lists, self.linear.named_parameters())

          return iter - self.dec, x, y #  why return x, y

        return _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, (iter, x, y), self.weight_bias_lists, self.bn_weight_bias_lists)

    linear_model = SimpleLinear()
    linear_model.to(device)
    #breakpoint()
    # input = torch.randn(2, 2).to(device)
    bs=16
    l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
    l_out = torch.randn(2, 2, dtype=torch.float32, device=device)
    iter = torch.tensor(3, device=device)
    res = linear_model(iter, l_in_0, l_out)
    print("res: ", res[-1])


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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
