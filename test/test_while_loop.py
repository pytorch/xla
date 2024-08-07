import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch_xla.experimental.fori_loop import fori_loop, _xla_while_loop_wrapper
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
import torch_xla.utils.utils as xu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import numpy as np
import time


def _fake_while_loop(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands

# def extend_bias_to_support_concatenate_arrays(bias, weights):
#   # check weights and bias has the same layer number
#   if size(weights) != size(bias):
#     print("weights and bias has not the same size!")


class WhileLoopTest(unittest.TestCase):

  def test_while_loop_addition(self):
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(x, 1)

    init_val = torch.tensor(3, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    _, res_with_loop = while_loop(cond_fn, body_fn, (iteri, init_val))
    _, res_without_loop = _fake_while_loop(cond_fn, body_fn, (iteri, init_val))
    self.assertTrue(torch.all(torch.eq(res_with_loop, res_without_loop)))

  def test_while_loop_addition_nested(self):
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(torch.add(x, 1), 1)

    init_val = torch.tensor(2, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    _, res_with_loop = while_loop(cond_fn, body_fn, (iteri, init_val))
    _, res_without_loop = _fake_while_loop(cond_fn, body_fn, (iteri, init_val))
    self.assertTrue(torch.all(torch.eq(res_with_loop, res_without_loop)))

  def test_while_loop_simple_linear_inside_loop(self):
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    class SimpleLinear(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

      def forward(self, iteri, x):

        def cond_fn(iteri, x):
          return iteri > 0

        def body_fn(iteri, x):
          return iteri - 1, self.linear(x)

        return while_loop(cond_fn, body_fn, (iteri, x))

      def forward_without_while_loop_op(self, iteri, x):
        while (iteri > 0):
          x = self.linear(x)
          iteri -= 1
        return iteri, x

    linear_model = SimpleLinear()
    linear_model.to(device)
    l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
    iteri = torch.tensor(10, dtype=torch.int32, device=device)
    _, res_with_loop = linear_model(iteri, l_in_0)
    _, res_without_loop = linear_model.forward_without_while_loop_op(
        iteri, l_in_0)

    self.assertTrue(torch.all(torch.eq(res_with_loop, res_without_loop)))

  def test_while_loop_simple_linear_outside_loop_change_weight_bias_benchmarking(self):
    start_time1 = time.time()
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
    weights = torch.tensor([[[5.1, 6.2], [7.3, 8.4]], [[1.0, 2.0], [3.0, 4.0]],
                            [[1.0, 2.0], [3.0, 4.0]]],
                           device=device)

    bias = torch.tensor([[[16.0, 17.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]],
                         [[1.0, 2.0], [0.0, 0.0]]],
                        device=device)

    def cond_fn(iteri, weights, bias, x):
      return iteri >= 0

    def body_fn(iteri, weights, bias, x):
      local_wieght_value = x[0]
      local_bias_value = x[1][0]
      x_val = x[2]
      local_linear = torch.nn.Linear(2, 2)
      local_linear.weight = torch.nn.parameter.Parameter(
          data=local_wieght_value, requires_grad=False)
      local_linear.bias = torch.nn.parameter.Parameter(
          data=local_bias_value, requires_grad=False)
      next_iteri = iteri - 1
      next_x = torch.stack((weights[-next_iteri - 1], bias[-next_iteri - 1],
                            local_linear(x_val)))
      return next_iteri, weights, bias, next_x

    inputs = torch.stack((weights[0], bias[0],
                          torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                       dtype=torch.float32,
                                       device=device)))
    print("inputs: ", inputs)  # needed to enable func catch stacked inputs
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    _, _, _, res = _xla_while_loop_wrapper(
        cond_fn, body_fn, (iteri, weights, bias, inputs), (),
        fake_tensor=False)  # need point out weight/bias in cond/body
    print("res: ", res)
    print("--- 3 layer while_loop: %s seconds ---" % (time.time() - start_time1))

    start_time2 = time.time()
    expected = inputs
    while (iteri >= 0):
      weight_value = expected[0]
      bias_value = expected[1][0]
      x = expected[2]
      local_linear_2 = torch.nn.Linear(2, 2)
      local_linear_2.weight = torch.nn.parameter.Parameter(
          data=weight_value, requires_grad=False)
      local_linear_2.bias = torch.nn.parameter.Parameter(
          data=bias_value, requires_grad=False)
      iteri = iteri - 1
      expected = torch.stack(
          (weights[-iteri - 1], bias[-iteri - 1], local_linear_2(x)))
    print("final expected: ", expected)
    print("--- 3 layers while: %s seconds ---" % (time.time() - start_time2))

    self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

  def test_while_loop_benchmark_while_loop(self):
    def benchmarking_only_while_loop():
      start_time1 = time.time()
      device = xm.xla_device()
      torch.set_grad_enabled(False)

      # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
      weights = torch.tensor([[[5.1, 6.2], [7.3, 8.4]], [[1.0, 2.0], [3.0, 4.0]],
                              [[1.0, 2.0], [3.0, 4.0]]],
                            device=device)

      bias = torch.tensor([[[16.0, 17.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]],
                          [[1.0, 2.0], [0.0, 0.0]]],
                          device=device)

      def cond_fn(iteri, weights, bias, x):
        return iteri >= 0

      def body_fn(iteri, weights, bias, x):
        local_wieght_value = x[0]
        local_bias_value = x[1][0]
        x_val = x[2]
        local_linear = torch.nn.Linear(2, 2)
        local_linear.weight = torch.nn.parameter.Parameter(
            data=local_wieght_value, requires_grad=False)
        local_linear.bias = torch.nn.parameter.Parameter(
            data=local_bias_value, requires_grad=False)
        next_iteri = iteri - 1
        next_x = torch.stack((weights[-next_iteri - 1], bias[-next_iteri - 1],
                              local_linear(x_val)))
        return next_iteri, weights, bias, next_x

      inputs = torch.stack((weights[0], bias[0],
                            torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                        dtype=torch.float32,
                                        device=device)))
      print("inputs: ", inputs)  # needed to enable func catch stacked inputs
      iteri = torch.tensor(2, dtype=torch.int32, device=device)
      _, _, _, res = _xla_while_loop_wrapper(
          cond_fn, body_fn, (iteri, weights, bias, inputs), (),
          fake_tensor=False)  # need point out weight/bias in cond/body
      print("res: ", res)
      print("--- 3 layer while_loop: %s seconds ---" % (time.time() - start_time1))

      # start_time2 = time.time()
      # expected = inputs
      # while (iteri >= 0):
      #   weight_value = expected[0]
      #   bias_value = expected[1][0]
      #   x = expected[2]
      #   local_linear_2 = torch.nn.Linear(2, 2)
      #   local_linear_2.weight = torch.nn.parameter.Parameter(
      #       data=weight_value, requires_grad=False)
      #   local_linear_2.bias = torch.nn.parameter.Parameter(
      #       data=bias_value, requires_grad=False)
      #   iteri = iteri - 1
      #   expected = torch.stack(
      #       (weights[-iteri - 1], bias[-iteri - 1], local_linear_2(x)))
      # print("final expected: ", expected)
      # print("--- 3 layers while: %s seconds ---" % (time.time() - start_time2))

      # self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

    for i in range(6):
      print("------------ ", i, " th -------------")
      # calculate time outside too
      benchmarking_only_while_loop()

  def test_while_loop_benchmark_while(self):
    def benchmarking_only_while():
      start_time2 = time.time()
      device = xm.xla_device()
      torch.set_grad_enabled(False)

      # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
      weights = torch.tensor([[[5.1, 6.2], [7.3, 8.4]], [[1.0, 2.0], [3.0, 4.0]],
                              [[1.0, 2.0], [3.0, 4.0]]],
                            device=device)

      bias = torch.tensor([[[16.0, 17.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]],
                          [[1.0, 2.0], [0.0, 0.0]]],
                          device=device)

      # def cond_fn(iteri, weights, bias, x):
      #   return iteri >= 0

      # def body_fn(iteri, weights, bias, x):
      #   local_wieght_value = x[0]
      #   local_bias_value = x[1][0]
      #   x_val = x[2]
      #   local_linear = torch.nn.Linear(2, 2)
      #   local_linear.weight = torch.nn.parameter.Parameter(
      #       data=local_wieght_value, requires_grad=False)
      #   local_linear.bias = torch.nn.parameter.Parameter(
      #       data=local_bias_value, requires_grad=False)
      #   next_iteri = iteri - 1
      #   next_x = torch.stack((weights[-next_iteri - 1], bias[-next_iteri - 1],
      #                         local_linear(x_val)))
      #   return next_iteri, weights, bias, next_x

      inputs = torch.stack((weights[0], bias[0],
                            torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                        dtype=torch.float32,
                                        device=device)))
      print("inputs: ", inputs)  # needed to enable func catch stacked inputs
      iteri = torch.tensor(2, dtype=torch.int32, device=device)
      # _, _, _, res = _xla_while_loop_wrapper(
      #     cond_fn, body_fn, (iteri, weights, bias, inputs), (),
      #     fake_tensor=False)  # need point out weight/bias in cond/body
      # print("res: ", res)
      # print("--- 3 layer while_loop: %s seconds ---" % (time.time() - start_time1))

      # start_time2 = time.time()
      expected = inputs
      while (iteri >= 0):
        weight_value = expected[0]
        bias_value = expected[1][0]
        x = expected[2]
        local_linear_2 = torch.nn.Linear(2, 2)
        local_linear_2.weight = torch.nn.parameter.Parameter(
            data=weight_value, requires_grad=False)
        local_linear_2.bias = torch.nn.parameter.Parameter(
            data=bias_value, requires_grad=False)
        iteri = iteri - 1
        expected = torch.stack(
            (weights[-iteri - 1], bias[-iteri - 1], local_linear_2(x)))
      print("final expected: ", expected)
      print("--- 3 layers while: %s seconds ---" % (time.time() - start_time2))

      # self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

    for i in range(6):
      print("------------ ", i, " th -------------")
      # calculate time outside too
      benchmarking_only_while()

  def test_while_loop_MNIST_benchmark(self):
    # device = xm.xla_device()
    torch.set_grad_enabled(False)

    class MNIST(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(10).eval()
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm2d(20).eval()
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.bnLayersWeights = []

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

          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

        return while_loop(cond_fn, body_fn, (iteri, x, y))

      def forward_compare(self, iteri, x, y):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        y = self.bn1(y)
        y = F.relu(F.max_pool2d(self.conv2(y), 2))
        y = self.bn2(y)
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

    mnist = MNIST()
    # mnist.to(device)
    l_in_0 = torch.randn(16, 1, 28, 28, dtype=torch.float32)
    l_out = torch.randn(16, 10, dtype=torch.float32)
    iteri = torch.tensor(10, dtype=torch.int64)

    start_time1 = time.time()
    _, _, res = mnist(iteri, l_in_0, l_out)
    print("res: ", res)
    print("--- res10: %s seconds ---" % (time.time() - start_time1))

    # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
    start_time2 = time.time()
    for i in range(iteri):
      _, _, expected = mnist.forward_compare(iteri, l_in_0, l_out)
    print("expected: ", expected)
    print("--- while res10: %s seconds ---" % (time.time() - start_time2))
    # print("res: ", res)
    # print("expected: ", expected)
    self.assertTrue(torch.all(torch.eq(res, expected)))

  def test_while_loop_MNIST_benchmark_while_loop_xla(self):
    def test_MNIST_PyTorch_XLA_2():
      device = xm.xla_device()
      torch.set_grad_enabled(False)

      class MNIST(torch.nn.Module):

        def __init__(self):
          super().__init__()
          self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
          self.bn1 = torch.nn.BatchNorm2d(10).eval()
          self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
          self.bn2 = torch.nn.BatchNorm2d(20).eval()
          self.fc1 = torch.nn.Linear(500, 50)
          self.fc2 = torch.nn.Linear(50, 10)
          self.bnLayersWeights = []

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

            return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

          return while_loop(cond_fn, body_fn, (iteri, x, y))

        def forward_compare(self, iteri, x, y):
          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)
          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

      mnist = MNIST()
      mnist.to(device)
      l_in_0 = torch.randn(16, 1, 28, 28, dtype=torch.float32, device=device)
      l_out = torch.randn(16, 10, dtype=torch.float32, device=device)
      iteri = torch.tensor(100000, dtype=torch.int64, device=device)

      start_time1 = time.time()
      _, _, res = mnist(iteri, l_in_0, l_out)
      print("res: ", res)
      print("--- res10: %s seconds ---" % (time.time() - start_time1))

      # # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
      # start_time2 = time.time()
      # _, _, expected = mnist.forward_compare(iteri, l_in_0, l_out)
      # print("expected: ", expected)
      # print("--- while res10: %s seconds ---" % (time.time() - start_time2))
      # # print("res: ", res)
      # # print("expected: ", expected)
      # self.assertTrue(torch.all(torch.eq(res, expected)))

    for i in range(6):
      print("------------ ", i, " th -------------")
      # calculate time outside too
      test_MNIST_PyTorch_XLA_2()

  def test_while_loop_MNIST_benchmark_while_xla(self):
    def test_MNIST_PyTorch_XLA_3():
      device = xm.xla_device()
      torch.set_grad_enabled(False)

      class MNIST(torch.nn.Module):

        def __init__(self):
          super().__init__()
          self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
          self.bn1 = torch.nn.BatchNorm2d(10).eval()
          self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
          self.bn2 = torch.nn.BatchNorm2d(20).eval()
          self.fc1 = torch.nn.Linear(500, 50)
          self.fc2 = torch.nn.Linear(50, 10)
          self.bnLayersWeights = []

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

            return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

          return while_loop(cond_fn, body_fn, (iteri, x, y))

        def forward_compare(self, iteri, x, y):
          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)
          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

      mnist = MNIST()
      mnist.to(device)
      l_in_0 = torch.randn(16, 1, 28, 28, dtype=torch.float32, device=device)
      l_out = torch.randn(16, 10, dtype=torch.float32, device=device)
      iteri = torch.tensor(100000, dtype=torch.int64, device=device)

      # start_time1 = time.time()
      # _, _, res = mnist(iteri, l_in_0, l_out)
      # print("res: ", res)
      # print("--- res10: %s seconds ---" % (time.time() - start_time1))

      # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
      start_time2 = time.time()
      _, _, expected = mnist.forward_compare(iteri, l_in_0, l_out)
      print("expected: ", expected)
      print("--- while res10: %s seconds ---" % (time.time() - start_time2))
      # print("res: ", res)
      # print("expected: ", expected)
      # self.assertTrue(torch.all(torch.eq(res, expected)))

    for i in range(6):
      print("------------ ", i, " th -------------")
      # calculate time outside too
      test_MNIST_PyTorch_XLA_3()

  def test_while_loop_MNIST_benchmark_while_loop(self):
    def MNIST_benchmark_while_loop():
      device = xm.xla_device()
      start_time1 = time.time()
      torch.set_grad_enabled(False)

      class MNIST(torch.nn.Module):

        def __init__(self):
          super().__init__()
          self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
          self.bn1 = torch.nn.BatchNorm2d(10).eval()
          self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
          self.bn2 = torch.nn.BatchNorm2d(20).eval()
          self.fc1 = torch.nn.Linear(500, 50)
          self.fc2 = torch.nn.Linear(50, 10)
          self.bnLayersWeights = []

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

            return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

          return while_loop(cond_fn, body_fn, (iteri, x, y))

        def forward_compare(self, iteri, x, y):
          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)
          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

      mnist = MNIST()
      mnist.to(device)
      l_in_0 = torch.randn(16, 1, 28, 28, dtype=torch.float32, device = device)
      l_out = torch.randn(16, 10, dtype=torch.float32)
      iteri = torch.tensor(1000000, dtype=torch.int64)

      # start_time1 = time.time()
      _, _, res = mnist(iteri, l_in_0, l_out)
      print("res: ", res)
      print("--- res10: %s seconds ---" % (time.time() - start_time1))

      # # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
      # start_time2 = time.time()
      # for i in range(iteri):
      #   _, _, expected = mnist.forward_compare(iteri, l_in_0, l_out)
      # print("expected: ", expected)
      # print("--- while res10: %s seconds ---" % (time.time() - start_time2))
      # # print("res: ", res)
      # # print("expected: ", expected)
      # self.assertTrue(torch.all(torch.eq(res, expected)))

    for i in range(6):
      print("------------ ", i, " th -------------")
      # calculate time outside too
      MNIST_benchmark_while_loop()

  def test_while_loop_MNIST_benchmark_while(self):
    def MNIST_benchmark_while():
      # device = xm.xla_device()
      start_time2 = time.time()
      torch.set_grad_enabled(False)

      class MNIST(torch.nn.Module):

        def __init__(self):
          super().__init__()
          self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
          self.bn1 = torch.nn.BatchNorm2d(10).eval()
          self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
          self.bn2 = torch.nn.BatchNorm2d(20).eval()
          self.fc1 = torch.nn.Linear(500, 50)
          self.fc2 = torch.nn.Linear(50, 10)
          self.bnLayersWeights = []

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

            return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

          return while_loop(cond_fn, body_fn, (iteri, x, y))

        def forward_compare(self, iteri, x, y):
          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y)
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)
          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

      mnist = MNIST()
      mnist.to(device)
      l_in_0 = torch.randn(16, 1, 28, 28, dtype=torch.float32)
      l_out = torch.randn(16, 10, dtype=torch.float32)
      iteri = torch.tensor(1000000, dtype=torch.int64)

      # # start_time1 = time.time()
      # _, _, res = mnist(iteri, l_in_0, l_out)
      # print("res: ", res)
      # print("--- res10: %s seconds ---" % (time.time() - start_time1))

      # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
      # start_time2 = time.time()
      for i in range(iteri):
        _, _, expected = mnist.forward_compare(iteri, l_in_0, l_out)
      print("expected: ", expected)
      print("--- pure while mnist1000: %s seconds ---" % (time.time() - start_time2))
      # print("res: ", res)
      # print("expected: ", expected)
      # self.assertTrue(torch.all(torch.eq(res, expected)))

    for i in range(6):
      print("------------ ", i, " th -------------")
      # calculate time outside too
      MNIST_benchmark_while()


  def test_while_loop_simple_linear_outside_loop_change_weight_bias_submit_clean(self):
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
    weights = torch.tensor([[[5.1, 6.2], [7.3, 8.4]], [[1.0, 2.0], [3.0, 4.0]],
                            [[1.0, 2.0], [3.0, 4.0]]],
                           device=device)

    bias = torch.tensor([[[16.0, 17.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]],
                         [[1.0, 2.0], [0.0, 0.0]]],
                        device=device)

    def cond_fn(iteri, weights, bias, x):
      return iteri >= 0

    def body_fn(iteri, weights, bias, x):
      local_wieght_value = x[0]
      local_bias_value = x[1][0]
      x_val = x[2]
      local_linear = torch.nn.Linear(2, 2)
      local_linear.weight = torch.nn.parameter.Parameter(
          data=local_wieght_value, requires_grad=False)
      local_linear.bias = torch.nn.parameter.Parameter(
          data=local_bias_value, requires_grad=False)
      next_iteri = iteri - 1
      next_x = torch.stack(
          (weights[-next_iteri-1], bias[-next_iteri-1], local_linear(x_val)))
      return next_iteri, weights, bias, next_x

    inputs = torch.stack((weights[0], bias[0],
                          torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                                       dtype=torch.float32,
                                       device=device)))
    print("inputs: ", inputs)  # needed to enable func catch stacked inputs
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    _, _, _, res = _xla_while_loop_wrapper(
        cond_fn, body_fn, (iteri, weights, bias, inputs), (),
        fake_tensor=False)  # need point out weight/bias in cond/body
    print("res: ", res)

    expected = inputs
    while (iteri >= 0):
      weight_value = expected[0]
      bias_value = expected[1][0]
      x = expected[2]
      local_linear_2 = torch.nn.Linear(2, 2)
      local_linear_2.weight = torch.nn.parameter.Parameter(
          data=weight_value, requires_grad=False)
      local_linear_2.bias = torch.nn.parameter.Parameter(
          data=bias_value, requires_grad=False)
      iteri = iteri - 1
      expected = torch.stack((weights[-iteri-1], bias[-iteri-1], local_linear_2(x)))
    print("final expected: ", expected)

    self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

  def test_while_loop_simple_linear_outside_loop_change_weight_bias_submit(self):
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
    weights = torch.tensor([[[5.1, 6.2], [7.3, 8.4]], [[1.0, 2.0], [3.0, 4.0]],
                            [[1.0, 2.0], [3.0, 4.0]]],
                           device=device)

    # bias = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]],
    #                      [[16.1, 17.2], [18.3, 19.4]]],
    #                     device=device)
    bias = torch.tensor([[[16.0, 17.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]],
                         [[1.0, 2.0], [0.0, 0.0]]],
                        device=device)
    # # original_bias = torch.tensor([[[16.1, 17.2],], [[1.0, 2.0],], [[1.0, 2.0],]], device=device)
    # # bias = torch.nn.functional.pad(original_bias, (0, 0, 0, 1), mode='constant')
    # bias = torch.nn.functional.pad(torch.tensor([[[16.1, 17.2],], [[1.0, 2.0],], [[1.0, 2.0],]], device=device), (0, 0, 0, 1), mode='constant')
    # # print("bias", bias)
    # print("bias", bias)
    # print("weights", weights)

    # iteri = torch.tensor(0, dtype=torch.int32, device=device)

    def cond_fn(iteri, weights, bias, x):
      return iteri >= 0
      # return iteri <= torch.tensor(2, dtype=torch.int32, device=device)

    def body_fn(iteri, weights, bias, x):
      local_wieght_value = x[0]
      local_bias_value = x[1][0]
      x_val = x[2]
      local_linear = torch.nn.Linear(2, 2)
      local_linear.weight = torch.nn.parameter.Parameter(
          data=local_wieght_value, requires_grad=False)
      local_linear.bias = torch.nn.parameter.Parameter(
          data=local_bias_value, requires_grad=False)
      next_iteri = iteri - 1
      # next_iteri = iteri + 1
      # add_dim_bias = (bias[-next_iteri]).unsqueeze(0) 
      # next_bias = torch.nn.functional.pad(add_dim_bias, (0, 0, 0, 1), mode='constant', value=0)
      next_x = torch.stack(
          (weights[-next_iteri-1], bias[-next_iteri-1], local_linear(x_val)))
      return next_iteri, weights, bias, next_x

    # inputs = torch.stack((weights[2], bias[2],
    #                       torch.tensor([[1.0, 1.0], [1.0, 1.0]],
    #                                    dtype=torch.float32,
    #                                    device=device)))
    inputs = torch.stack((weights[0], bias[0],
                          torch.tensor([[1.0, 1.0], [1.0, 1.0]],
                                       dtype=torch.float32,
                                       device=device)))
    print("inputs: ", inputs)  # needed to enable func catch stacked inputs
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    # iteri = torch.tensor(0, dtype=torch.int32, device=device)
    _, _, _, res = _xla_while_loop_wrapper(
        cond_fn, body_fn, (iteri, weights, bias, inputs), (),
        fake_tensor=False)  # need point out weight/bias in cond/body
    print("res: ", res)

    # print("bias", bias)
    # print("weights", weights)
    # print("inputs: ", inputs)
    expected = inputs
    while (iteri >= 0):
    # while (iteri <= 2):
      weight_value = expected[0]
      bias_value = expected[1][0]
      x = expected[2]
      local_linear_2 = torch.nn.Linear(2, 2)
      local_linear_2.weight = torch.nn.parameter.Parameter(
          data=weight_value, requires_grad=False)
      local_linear_2.bias = torch.nn.parameter.Parameter(
          data=bias_value, requires_grad=False)
      iteri = iteri - 1
      # iteri = iteri + 1
      expected = torch.stack((weights[-iteri-1], bias[-iteri-1], local_linear_2(x)))
    print("final expected: ", expected)

    self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

  def test_while_loop_simple_linear_outside_loop_change_weight_bias_ole(self):
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
    weights = torch.tensor([[[5.1, 6.2], [7.3, 8.4]], [[1.0, 2.0], [3.0, 4.0]],
                            [[1.0, 2.0], [3.0, 4.0]]],
                           device=device)

    # bias = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]],
    #                      [[16.1, 17.2], [18.3, 19.4]]],
    #                     device=device)
    bias = torch.tensor([[[16.0, 17.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]],
                         [[1.0, 2.0], [0.0, 0.0]]],
                        device=device)
    # original_bias = torch.tensor([[[1.0, 2.0],], [[1.0, 2.0],], [[16.1, 17.2],]], device=device)
    # bias = torch.nn.functional.pad(original_bias, (0, 0, 0, 1), mode='constant')
    # print("bias", bias)
    print("bias", bias)
    print("weights", weights)

    # iteri = torch.tensor(0, dtype=torch.int32, device=device)

    def cond_fn(iteri, weights, bias, x):
      return iteri >= 0
      # return iteri <= torch.tensor(2, dtype=torch.int32, device=device)

    def body_fn(iteri, weights, bias, x):
      local_wieght_value = x[0]
      local_bias_value = x[1][0]
      x_val = x[2]
      local_linear = torch.nn.Linear(2, 2)
      local_linear.weight = torch.nn.parameter.Parameter(
          data=local_wieght_value, requires_grad=False)
      local_linear.bias = torch.nn.parameter.Parameter(
          data=local_bias_value, requires_grad=False)
      next_iteri = iteri - 1
      # next_iteri = iteri + 1
      # add_dim_bias = (bias[-next_iteri]).unsqueeze(0) 
      # next_bias = torch.nn.functional.pad(add_dim_bias, (0, 0, 0, 1), mode='constant', value=0)
      next_x = torch.stack(
          (weights[-next_iteri-1], bias[-next_iteri-1], local_linear(x_val)))
      return next_iteri, weights, bias, next_x

    # inputs = torch.stack((weights[2], bias[2],
    #                       torch.tensor([[1.0, 1.0], [1.0, 1.0]],
    #                                    dtype=torch.float32,
    #                                    device=device)))
    inputs = torch.stack((weights[0], bias[0],
                          torch.tensor([[1.0, 1.0], [1.0, 1.0]],
                                       dtype=torch.float32,
                                       device=device)))
    print("inputs: ", inputs)  # needed to enable func catch stacked inputs
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    # iteri = torch.tensor(0, dtype=torch.int32, device=device)
    _, _, _, res = _xla_while_loop_wrapper(
        cond_fn, body_fn, (iteri, weights, bias, inputs), (),
        fake_tensor=False)  # need point out weight/bias in cond/body
    print("res: ", res)

    print("bias", bias)
    print("weights", weights)
    print("inputs: ", inputs)
    expected = inputs
    while (iteri >= 0):
    # while (iteri <= 2):
      weight_value = expected[0]
      bias_value = expected[1][0]
      x = expected[2]
      local_linear_2 = torch.nn.Linear(2, 2)
      local_linear_2.weight = torch.nn.parameter.Parameter(
          data=weight_value, requires_grad=False)
      local_linear_2.bias = torch.nn.parameter.Parameter(
          data=bias_value, requires_grad=False)
      iteri = iteri - 1
      # iteri = iteri + 1
      expected = torch.stack((weights[-iteri-1], bias[-iteri-1], local_linear_2(x)))
    print("final expected: ", expected)

    self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

  def test_while_loop_simple_linear_outside_loop_change_weight_bias(self):
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    # TODO(@manfei): enable weights[0] != weights[1] and bias[0] != bias[1], now test pass with weights[0] == weights[1] and bias[0]==bias[1]
    weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]],
                            [[5.1, 6.2], [7.3, 8.4]]],
                           device=device)

    # bias = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]],
    #                      [[16.1, 17.2], [18.3, 19.4]]],
    #                     device=device)
    original_bias = torch.tensor([[[16.1, 17.2],], [[1.0, 2.0],], [[1.0, 2.0],]], device=device)
    bias = torch.nn.functional.pad(original_bias, (0, 0, 0, 1), mode='constant')

    def cond_fn(iteri, weights, bias, x):
      return iteri >= 0

    def body_fn(iteri, weights, bias, x):
      local_wieght_value = x[0]
      local_bias_value = x[1][0]
      x_val = x[2]
      local_linear = torch.nn.Linear(2, 2)
      local_linear.weight = torch.nn.parameter.Parameter(
          data=local_wieght_value, requires_grad=False)
      local_linear.bias = torch.nn.parameter.Parameter(
          data=local_bias_value, requires_grad=False)
      next_iteri = iteri - 1
      next_x = torch.stack(
          (weights[-next_iteri-1], bias[-next_iteri-1], local_linear(x_val)))
      return next_iteri, weights, bias, next_x

    inputs = torch.stack((weights[2], bias[2],
                          torch.tensor([[1.0, 1.0], [1.0, 1.0]],
                                       dtype=torch.float32,
                                       device=device)))
    print("inputs: ", inputs)  # needed to enable func catch stacked inputs
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    _, _, _, res = _xla_while_loop_wrapper(
        cond_fn, body_fn, (iteri, weights, bias, inputs), (),
        fake_tensor=False)  # need point out weight/bias in cond/body
    print("res: ", res)

    expected = inputs
    while (iteri >= 0):
      weight_value = expected[0]
      bias_value = expected[1][0]
      x = expected[2]
      local_linear_2 = torch.nn.Linear(2, 2)
      local_linear_2.weight = torch.nn.parameter.Parameter(
          data=weight_value, requires_grad=False)
      local_linear_2.bias = torch.nn.parameter.Parameter(
          data=bias_value, requires_grad=False)
      iteri = iteri - 1
      expected = torch.stack((weights[-iteri-1], bias[-iteri-1], local_linear_2(x)))
    print("final expected: ", expected)

    self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

  def test_while_loop_simple_linear_inside_loop_change_weight_bias(self):
    device = xm.xla_device()
    torch.set_grad_enabled(False)

    class SimpleLinear(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

        self.weights = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]],
             [[5.1, 6.2], [7.3, 8.4]]],
            device=device)

        self.bias = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]],
             [[16.1, 17.2], [18.3, 19.4]]],
            device=device)

      def forward(self, iteri, x):

        def cond_fn(iteri, weights, bias, x):
          return iteri >= 0

        def body_fn(iteri, weights, bias, x):
          local_wieght_value = x[0]
          local_bias_value = x[1]
          x_val = x[2]
          local_linear = torch.nn.Linear(2, 2)
          local_linear.weight = torch.nn.parameter.Parameter(
              data=local_wieght_value, requires_grad=False)
          local_linear.bias = torch.nn.parameter.Parameter(
              data=local_bias_value, requires_grad=False)
          next_iteri = iteri - 1
          next_x = torch.stack(
              (weights[next_iteri], bias[next_iteri], local_linear(x_val)))
          return next_iteri, weights, bias, next_x

        return _xla_while_loop_wrapper(
            cond_fn,
            body_fn, (iteri, self.weights, self.bias, x), (),
            fake_tensor=False)

      def forward_compare(self, iteri, x):
        while (iteri >= 0):
          weight_value = x[0]
          bias_value = x[1]
          x_val = x[2]
          local_linear_2 = torch.nn.Linear(2, 2)
          local_linear_2.weight = torch.nn.parameter.Parameter(
              data=weight_value, requires_grad=False)
          local_linear_2.bias = torch.nn.parameter.Parameter(
              data=bias_value, requires_grad=False)
          iteri = iteri - 1
          x = torch.stack(
              (self.weights[iteri], self.bias[iteri], local_linear_2(x_val)))
        return iteri, x

    linear_model = SimpleLinear()
    linear_model.to(device)
    inputs = torch.stack((linear_model.weights[2], linear_model.bias[2],
                          torch.tensor([[1.0, 1.0], [1.0, 1.0]],
                                       dtype=torch.float32,
                                       device=device)))
    print("inputs: ", inputs)
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    iter_value, _, _, res = linear_model(iteri, inputs)
    print("res: ", res)

    # === expected result after 2 iteration to be compared ===
    test_value = inputs
    _, test_value = linear_model.forward_compare(iteri, test_value)
    expected = test_value
    print("expected: ", expected)

    self.assertTrue(torch.all(torch.eq(res[2], expected[2])))

  # ====== fori_loop ======
  @unittest.skip("Fori_loop is not supported now due to unstable result.")
  def test_fori_loop_addition(self):
    device = xm.xla_device()

    lower = torch.tensor(0, device=device)
    upper = torch.tensor(50, device=device)
    init_val = torch.tensor(1, dtype=torch.int32, device=device)

    def body_fun(x):
      return torch.add(x, 1)

    _, res_with_loop = fori_loop(lower, upper, body_fun, (init_val))

    # === expected ===
    for i in range(upper - lower):
      init_val = torch.add(init_val, 1)
    res_without_loop = init_val


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
