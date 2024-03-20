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


def fori_loop(lower, upper, body_fun, *init_vals): # *init_val):

  device = xm.xla_device()
  limit_value = upper
  init = lower
  iterator = lower

  # one_value_original = torch.tensor([1], dtype=torch.int32, device=device)

  def cond_fn(upper, lower, *init_vals):
    # init_val_compy = init_val.clone()
    one_value1 = torch.tensor([0], dtype=torch.int32, device=device)
    one_value2 = torch.tensor([0], dtype=torch.int32, device=device)
    lower = torch.add(lower, one_value1[0])
    lower = torch.sub(lower, one_value2[0])
    return lower[0] <= upper[0]

  def body_fn(upper, lower, *init_vals):
    # one_value_original = torch.tensor(1, dtype=torch.int32, device=device)
    # (a, b) = init_vals
    # return (upper, torch.add(lower, 1), body_fun(a, b), b.clone())
    return (upper, torch.add(lower, 1), body_fun(*init_vals), init_vals[1]) # init_vals[1:])
    # (body_fun(*init_vals)).clone(), init_vals[1].clone())
    # body_fun(one_value_original, init_val)) # body_fun(lower, init_val))

  res = while_loop(cond_fn, body_fn, (upper, lower, *init_vals))
  # res = _xla_while_loop(cond_fn, body_fn, (upper, lower, *init_vals))
  # print("upper: ", upper)
  # print("lower: ", lower)
  return res

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
