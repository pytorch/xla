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
  (plus_value, init_val) = init_val
  for i in range((upper - lower)[0]):
    plus_value, init_val = body_fun(plus_value, init_val)
  return init_val


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

  def test_fori_loop_tpu_addition(self):

    xm.mark_step()
    device = xm.xla_device()

    lower = torch.tensor([2], dtype=torch.int32, device=device)
    upper = torch.tensor([52], dtype=torch.int32, device=device)
    plus_value = torch.tensor([1], dtype=torch.int32, device=device)
    init_val = torch.tensor([1], dtype=torch.int32, device=device)

    def body_fun(*argus):
      plus_value, init_val = argus
      return plus_value, torch.add(plus_value, init_val)

    _, _, _, actual = fori_loop(upper, lower, body_fun, plus_value, init_val)
    expected = _fake_fori_loop(lower, upper, body_fun, plus_value, init_val)
    self.assertEqual(expected, actual)

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
    if res_xla_computation == null:
      print("print computation from _get_xla_computation: null !!!!!!!!!!!!!")
    else:
      hlo_print = xb.get_computation_hlo(res_xla_computation)
      print("print computation from _get_xla_computation: !!!!!!!!!")
      print(hlo_print)

    # ### test to see what _get_stablehlo would get
    # # tensors, torch_xla._XLAC._xla_get_default_device(), [],
    # res_hlo_xla_computation = torch_xla._XLAC._get_stablehlo([t3], torch_xla._XLAC._xla_get_default_device(), [], False)
    # print("res hlo: ", res_hlo_xla_computation)

    # ### test to get xlacomputation
    # body_result0 = torch.add(t1, t2)
    # body_ctx0 = torch_xla._XLAC.lowering.LoweringContext()
    # body_ctx0.build(list(body_result0))
    # body_hlo0 = body_ctx0.hlo()
    # print("body_hlo0 finish !!!")
    # print("type body_hlo0: ", type(body_hlo0))
    # # print("body_hlo0: ", body_hlo0)

    # ### test to use _get_stablehlo to get xlacomputation
    # body_result = torch.add(t1, t2)
    # # body_ctx = torch_xla._XLAC.lowering.LoweringContext()
    # body_stable_hlo = torch_xla._XLAC._get_stablehlo([t3], torch_xla._XLAC._xla_get_default_device(), [], False)
    # print("body_stable_hlo finish !!!")
    # print("type body_stable_hlo: ", type(body_stable_hlo))
    # print("body_stable_hlo: ", body_stable_hlo)

    # if body_hlo0 == body_stable_hlo:
    #   print("hlo and stablehlo are the same iteam")
    # else:
    #   print("hlo and stablehlo are not the same iteam")
    # # body_ctx.set_name_string("bodyctx")
    # # body_ctx.buildforiloop(list(body_result), [])
    # # body_hlo = body_ctx.hlo()
    # body_computation = xb.computation_from_module_proto("bodycomputation",
    #                                                     body_stable_hlo)
    # hlo_print = xb.get_computation_hlo(body_computation)
    # print("print computation from _get_stablehlo: !!!!!!!!!")
    # print(hlo_print)

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
    #device = ''
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, x):
        x = self.linear(x)
        return x
          # def cond_fn(it, x):
          #     return it - self.dec > 0

          # def body_fn(it, x):
          #     return it - 1, self.linear(x)

          # return while_loop(cond_fn, body_fn, (iter, x))

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    #breakpoint()
    input = torch.randn(2, 2).to(device)
    # iter = torch.tensor(3, device=device)
    # res = simple_with_linear(iter, input)
    t3 = simple_with_linear(input)
    # t1 = torch.randn(20, 5).to(device)
    # t2 = torch.randn(20, 5).to(device)
    # t3 = torch.add(t1, t2)

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

  def test_while_loop_get_xlacomputation_tpu_simple_linear_while_loop(self):

    xm.mark_step()
    device = xm.xla_device()
    #device = ''
    torch.set_grad_enabled(False)

    class SimpleWithLinear(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("dec", torch.tensor(1))

      def forward(self, x):
        x = self.linear(x)
        return x
          # def cond_fn(it, x):
          #     return it - self.dec > 0

          # def body_fn(it, x):
          #     return it - 1, self.linear(x)

          # return while_loop(cond_fn, body_fn, (iter, x))

    simple_with_linear = SimpleWithLinear()
    simple_with_linear.to(device)
    #breakpoint()
    input = torch.randn(2, 2).to(device)
    # iter = torch.tensor(3, device=device)
    # res = simple_with_linear(iter, input)
    t3 = simple_with_linear(input)
    # t1 = torch.randn(20, 5).to(device)
    # t2 = torch.randn(20, 5).to(device)
    # t3 = torch.add(t1, t2)

    def cond_fn(upper, lower, one_value, x, input_value, output_value, *args):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value, *args):
      new_lower = torch.add(one_value, lower)
      output_value = simple_with_linear(input_value)
      res = [upper.clone(), new_lower.clone(), one_value.clone(), torch.add(one_value, x), input_value.clone(), output_value.clone()]
      return tuple(res)
      # bn_list = []
      # for name, param in simple_with_linear.named_parameters():
      #   if name[:2]=='bn':
      #     bn_list.append(param)

      #   res.insert(-1, param)

      # # add still exist bn_list if the last additional_inputs is bn- pre
      # # add at the tile
      # if len(bn_list) !=0:
      #   output_value = res[-1]
      #   bn_list.reverse()
      #   res = res[:-1] + bn_list
      #   res.append(output_value)
      #   bn_list = []
      # return tuple(res)

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
    # body_result = body_fn(*fake_carried_inputs)
    body_ctx = torch_xla._XLAC.lowering.LoweringContext()
    body_ctx.set_name_string("bodyctx")
    # body_ctx.buildforiloop(list(body_result), [])
    body_ctx.buildforiloop(list(t3), [])
    body_hlo = body_ctx.hlo()
    body_computation = xb.computation_from_module_proto("bodycomputation",
                                                        body_hlo)
    body_hlo_print = xb.get_computation_hlo(body_computation)
    print("print computation from PyLoweringContext: !!!!!!!!!")
    print(body_hlo_print)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
