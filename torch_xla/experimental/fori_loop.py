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

def fori_loop(lower, upper, body_fun, one_value, init_val):

  device = xm.xla_device()

  def cond_fn(upper, lower, x):
    return lower[0] < upper[0]

  def body_fn(upper, lower, x):
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    return (torch.sub(upper, one_value), lower, body_fun(one_value, x))

  def old_cond_fn(one_value, lower, upper, init_val): 
    lower_compare = torch.add(lower, one_value)
    return lower_compare[0] <= upper[0]

  def old_body_fn(one_value, lower, upper, init_val):
    new_lower = torch.add(lower, one_value)
    new_init_val = body_fun(init_val, one_value)
    return (one_value, new_lower, upper, new_init_val)

  res = _xla_while_loop(cond_fn, body_fn, lower, upper, init_val)
  return res

@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  return _xla_while_loop(cond_fn, body_fn, operands)

def _xla_while_loop(cond_fn, body_fn, *operands):
  kwargs = {}
  shapes = xb.tensor_shape((operands))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # generate cond_fn xlacomputation
  cond_result = cond_fn(*operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  cond_ctx.build([cond_result], list(operands[2:]))
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)

  # generate body_fn xlacomputation
  body_result = body_fn(*operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.build(list(body_result), [])
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

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', (operands), computation)

  return result
