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

def _fake_while_loop_second(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands

def _fake_while_loop(cond_fn, body_fn, operands):
  # operands need to be more than one here
  while cond_fn(*operands):
    operands = body_fn(*operands)
  return operands


# test class
class WhileLoopTest(unittest.TestCase):

  # passed: torch pure version: subtraction
  def test_while_loop_tpu_subtraction_pure_torch(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x, y):
      return iteri > 0

    def body_fn(iteri, x, y):
      return iteri - 1, x, torch.sub(x, 1)

    init_val = torch.tensor(10)
    out_val = torch.tensor(15)
    iteri = torch.tensor(3)
    res = while_loop(cond_fn, body_fn, (iteri, init_val, out_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val, out_val))
    print("expected: ", expected)
    self.assertEqual(res, expected)

  # passed: torch_xla version: subtraction
  def test_while_loop_tpu_subtraction(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.sub(x, 1)

    init_val = torch.tensor(10, dtype=torch.int32, device=device)
    iteri = torch.tensor(3, device=device)
    res = while_loop(cond_fn, body_fn, (iteri, init_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val))
    print("expected: ", expected)
    self.assertEqual(res, expected) # order of res and expected matter

  # passed: torch pure version: addition
  def test_while_loop_tpu_addition_pure_torch(self):
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
    self.assertEqual(res, expected)

  # passed: torch_xla version: addition
  def test_while_loop_tpu_addition(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x): # make sure name of variable match
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(x, 1)

    init_val = torch.tensor(3, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    res =  while_loop(cond_fn, body_fn, (iteri, init_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val))
    print("expected: ", expected)
    self.assertEqual(res, expected) # order of res and expected matter

  # passed:  torch pure version: nestes addition
  def test_while_loop_tpu_addition_nested_pure_torch(self):

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
    self.assertEqual(res, expected) # order of res and expected matter

  # passed: torch_xla version: nestes subtraction
  def test_while_loop_tpu_addition_nested(self):
    xm.mark_step()
    device = xm.xla_device()

    def cond_fn(iteri, x):
      return iteri > 0

    def body_fn(iteri, x):
      return iteri - 1, torch.add(torch.add(x, 1), 1)

    init_val = torch.tensor(2, dtype=torch.int32, device=device)
    iteri = torch.tensor(10, device=device)
    res =  while_loop(cond_fn, body_fn, (iteri, init_val))
    print("res: ", res)
    expected = _fake_while_loop_second(cond_fn, body_fn, (iteri, init_val))
    print("expected: ", expected)
    # self.assertEqual(res, expected) # order of res and expected matter

  # torch_xla version: linear
  def test_while_loop_tpu_simple_linear_inside_loop(self):
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
    print("l_in_0: ", l_in_0)
    iteri = torch.tensor(3, dtype=torch.int32, device=device)
    A, res = linear_model(iteri, l_in_0)
    print("---------------------------------------------------------")
    print("A: ", A)
    print("res: ", res)
    print("---------------------------------------------------------")
    _, expected = linear_model.forward_compare(iteri, l_in_0)
    print("expected: ", expected)
    print("---------------------------------------------------------")
    print("l_in_0: ", l_in_0)
    print("---------------------------------------------------------")

  # torch_xla version: MNIST without bn layer
  def test_while_loop_tpu_MNIST_inside_loop(self):
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

      def forward(self, iteri, x, y):
        def cond_fn(iteri, x, y):
          return iteri > 0

        def body_fn(iteri, x, y):
          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          # z = self.bn1(z) # torch.while_loop's body_fn might be modifying the input!
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          # y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

        return while_loop(cond_fn, body_fn, (iteri, x, y))

      def forward_compare(self, iteri, x, y):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        # z = self.bn1(z) # torch.while_loop's body_fn might be modifying the input!
        y = F.relu(F.max_pool2d(self.conv2(y), 2))
        # y = self.bn2(y)
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

    mnist = MNIST()
    mnist.to(device)
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32, device=device)
    l_out = torch.randn(bs, 10, dtype=torch.float32, device=device)
    iteri = torch.tensor(3, dtype=torch.int64, device=device)
    _, _, res = mnist(iteri, l_in_0, l_out)
    print("res[0]: ", res[0])
    _, _, expected_res = mnist.forward_compare(iteri, l_in_0, l_out)
    print("expected_res[0]: ", res[0])

  # ====== test _get_xla_computation ======
  def test__get_xlacomputation(self):

    xm.mark_step()
    device = xm.xla_device()
    t1 = torch.randn(20, 5).to(device)
    t2 = torch.randn(20, 5).to(device)
    t3 = torch.add(t1, t2)

    ### implement one new function for xlacomputation generation with post-order
    res_xla_computation = torch_xla._XLAC._get_xla_computation([t3], [], True)
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("Gain and print computation from _get_xla_computation:")
      print(hlo_print)
    else:
      print("Failed to gain or print computation(null) from _get_xla_computation")

  def test_PyLoweringContext_and__get_xlacomputation_with_simple_linear_tpu(self):

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
    res_xla_computation = torch_xla._XLAC._get_xla_computation([t3], [], True)
    if res_xla_computation:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("Gain and print computation from _get_xla_computation:")
      print(hlo_print)
    else:
      print("Failed to gain or print computation(null) from _get_xla_computation")

    ### get xlacomputation via PyLoweringContext
    body_ctx = torch_xla._XLAC.lowering.LoweringContext()
    body_ctx.set_name_string("bodyctx")
    body_ctx.buildforiloop(list(t3), [])
    body_hlo = body_ctx.hlo()
    body_computation = xb.computation_from_module_proto("bodycomputation",
                                                        body_hlo)
    body_hlo_print = xb.get_computation_hlo(body_computation)
    print("Gain and print computation from PyLoweringContext:")
    print(body_hlo_print)

  # ====== fori_loop ======
  def test_fori_loop_addition_tpu(self):
    xm.mark_step()
    device = xm.xla_device()

    lower = torch.tensor(0, device=device)
    upper = torch.tensor(50, device=device)
    init_val = torch.tensor(1, dtype=torch.int32, device=device)

    def body_fun(x):
      return torch.add(x, 1)

    _, actual = fori_loop(upper, lower, body_fun, (init_val))
    print("actual: ", actual)

    # === expected ===
    x = init_val
    for i in range(upper - lower):
      x = torch.add(x, 1)
    expected = x
    print("expected: ", expected)

    # === compare actual result and expected result
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
