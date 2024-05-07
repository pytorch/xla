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


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)