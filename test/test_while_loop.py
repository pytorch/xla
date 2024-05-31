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
import torch_xla.utils.utils as xu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _fake_while_loop(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands


class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu_addition(self):
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(x, 1)

    init_val = torch.tensor(3, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    _, res = while_loop(cond_fn, body_fn, (iteri, init_val))
    _, expected = _fake_while_loop(cond_fn, body_fn, (iteri, init_val))
    self.assertTrue(torch.all(torch.eq(res, expected)))

  def test_while_loop_tpu_addition_nested(self):
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(torch.add(x, 1), 1)

    init_val = torch.tensor(2, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    _, res = while_loop(cond_fn, body_fn, (iteri, init_val))
    _, expected = _fake_while_loop(cond_fn, body_fn, (iteri, init_val))
    self.assertTrue(torch.all(torch.eq(res, expected)))

  def test_while_loop_tpu_simple_linear_inside_loop(self):
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

      def forward_compare(self, iteri, x):
        y = self.linear(x)
        return iteri - 1, y

    linear_model = SimpleLinear()
    linear_model.to(device)
    l_in_0 = torch.randn(2, 2, dtype=torch.float32, device=device)
    iteri = torch.tensor(2, dtype=torch.int32, device=device)
    _, res = linear_model(iteri, l_in_0)

    # === expected result after 2 iteration to be compared ===
    _, expected = linear_model.forward_compare(iteri, l_in_0)
    _, expected = linear_model.forward_compare(iteri, expected)

    self.assertTrue(torch.all(torch.eq(res, expected)))

  def test_while_loop_tpu_MNIST_inside_loop(self):
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
    iteri = torch.tensor(3, dtype=torch.int64, device=device)
    _, _, res = mnist(iteri, l_in_0, l_out)

    # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
    _, _, expected_res = mnist.forward_compare(iteri, l_in_0, l_out)
    self.assertTrue(torch.all(torch.eq(res, expected_res)))

  # ====== fori_loop ======
  def test_fori_loop_addition_tpu(self):
    device = xm.xla_device()

    lower = torch.tensor(0, device=device)
    upper = torch.tensor(50, device=device)
    init_val = torch.tensor(1, dtype=torch.int32, device=device)

    def body_fun(x):
      return torch.add(x, 1)

    _, res = fori_loop(upper, lower, body_fun, (init_val))

    # === expected ===
    x = init_val
    for i in range(upper - lower):
      x = torch.add(x, 1)
    expected = x
    print("res: ", res)
    print("expected: ", expected)
    self.assertTrue(torch.all(torch.eq(res, expected)))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
