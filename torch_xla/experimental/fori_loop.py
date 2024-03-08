import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
import torch._higher_order_ops.while_loop
from torch._higher_order_ops.while_loop import while_loop_op


def fori_loop(lower, upper, body_fun, *init_val):
  # example data:
  # init_val = torch.tensor([0], dtype=torch.int32, device=device)
  # lower = torch.tensor([0], dtype=torch.int32, device=device)
  # upper = torch.tensor([10], dtype=torch.int32, device=device)
  # limit_range = upper - lower
  device = xm.xla_device()
  # one_value = torch.tensor([0], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  # lower, upper
  limit_value = upper
  init = lower

  # device = xm.xla_device()
  def cond_fn(init, limit_value, *init_val):
    one_value = torch.tensor([1], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
    one_value_2 = torch.tensor([1], dtype=torch.int32, device=device)
    torch.add(one_value, one_value_2)
    a = torch.add(init_val[0], one_value)
    b = torch.add(init_val[1], one_value_2)
    # c = limit_value[0] >= init[0] & a==b
    return (limit_value[0] <= init[0]) and (a[0] >= b[0])

  def body_fn(init, limit_value, *init_val):
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    return (torch.add(init, one_value), limit_value.clone(), body_fun(*init_val), init_val[1])

    # TODO(@manfei): init and limit_value has to be torch.tensor.
    # init = torch.tensor([0], dtype=torch.int32, device=device)
    # limit_value = torch.tensor([10], dtype=torch.int32, device=device)
  res = while_loop(cond_fn, body_fn, (init, limit_value, *init_val))
    # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
    # self.assertEqual(expected, res)
  return res

  # def cond_fn(lower, upper, *init_val):
  #   one_value = torch.tensor([1], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  #   one_value_2 = torch.tensor([0], dtype=torch.int32, device=device)
  #   lower = torch.add(lower, one_value)
  #   lower = torch.add(lower, one_value_2)
  #   # init_val = init_val.clone()
  #   # print("lower: ",lower)
  #   return lower[0] <= upper[0]
  
  # def body_fn(lower, upper, *init_val):
  #   # one_value = torch.tensor([0], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  #   # lower = torch.add(lower, one_value)
  #   init_val_local = body_fun(*init_val)
  #   return (lower.clone(), upper.clone(), init_val_local, init_val[1])

  # a = while_loop(cond_fn, body_fn, (lower, upper, *init_val))
  # # print("result: finall: ", a)
  # return a # while_loop(cond_fn, body_fn, (lower, upper, init_val))


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)


def _xla_while_loop(cond_fn, body_fn, operands):

  # create inputs placeholder
  kwargs = {}
  shapes = xb.tensor_shape(operands)
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # generate cond_fn xlacomputation
  cond_result = cond_fn(*operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  cond_ctx.build([cond_result])
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)

  # generate body_fn xlacomputation
  body_result = body_fn(*operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.build(list(body_result))
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)
  hlo_print = xb.get_computation_hlo(computation)
  print("while computation: !!!!!!!!!")
  print(hlo_print)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 tuple(operands), computation)

  return result
