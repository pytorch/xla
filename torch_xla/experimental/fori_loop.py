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
import torch_xla.debug.metrics as met

# def fori_loop(lower, upper, body_fun, init_val):
#   val = init_val
#   for i in range(lower, upper):
#     val = body_fun(i, val)
#   return val

###
    # def cond_fn(x):
    #   ten = torch.ones(1, dtype=torch.int32, device=device)
    #   return x[0] >= ten[0]

    # def body_fn(x):
    #   return (torch.sub(x[0], 1),)

    # xi = torch.tensor([5], dtype=torch.int32, device=device)
    # res = while_loop(cond_fn, body_fn, (xi,))

def fori_loop(lower, upper, body_fun, init_val):

  device = xm.xla_device()
  upper_placeholder = torch.ones(1, dtype=torch.int32, device=device)
  upper_placeholder[0] = upper

  lower_placeholder = torch.ones(1, dtype=torch.int32, device=device)
  lower_placeholder[0] = lower

  iterator = lower_placeholder

  def cond_fn(operands): # iterator, init_val):
    # return iterator[0] <= upper_placeholder[0]
    # print("operands: ", operands)
    # print("type operands: ", type(operands))
    # print("operands[0]: ", operands[0])
    # print("type operands[0]: ", type(operands[0]))
    # print("operands[1]: ", operands[1])
    # print("type operands[1]: ", type(operands[1]))
    # print("operands[0][0]: ", operands[0][0])
    # print("type operands[0][0]: ", type(operands[0][0]))
    # print("operands[1][0]: ", operands[1][0])
    # print("type operands[1][0]: ", type(operands[1][0]))

    return operands[0] <= operands[1]

  def body_fn(operands): # iterator, init_val):
    # iterator[0] = iterator[0] - 1 # one = torch.ones(1, dtype=torch.int32, device=device) torch.sub(iterator[0] - one)
    # return body_fun(iterator, init_val)
    operands[0][0] = iterator[0] - 1 # one = torch.ones(1, dtype=torch.int32, device=device) torch.sub(iterator[0] - one)
    return body_fun(operands[0], operands[1])

  return while_loop(cond_fn, body_fn, (iterator, init_val))

@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)


def _xla_while_loop(cond_fn, body_fn, operands):

  # print("operands: ", operands)
  # print("type oeprands: ", type(operands))
  # create inputs placeholder
  kwargs = {}
  shapes = xb.tensor_shape(operands)
  builder = xb.create_builder('test_while')
  params = []
  # secondparams = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)
    # secondparams.append(xb.Op.tuple([p]))

  # generate cond_fn xlacomputation
  xm.mark_step()
  cond_result = cond_fn(operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.setnamestring("condctx")
  cond_ctx.build_for_while(list(cond_result))
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond_hlo: !!!!!!!!!")
  # print(cond_hlo_print)

  # generate body_fn xlacomputation
  xm.mark_step()
  body_result = body_fn(operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.setnamestring("bodyctx")
  body_ctx.build_for_while(list(body_result))
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  # body_hlo_print = xb.get_computation_hlo(body_computation)
  # print("body_hlo: !!!!!!!!!")
  # print(body_hlo_print)

  # create xla:While op with cond_computation and body_computation
  print("params: ", params)
  print("type params: ", type(params))
  input_tuple = xb.Op.tuple(params)
  print("input_tuple: ", input_tuple)
  print("type input_tuple: ", type(input_tuple))
  aaa_tuple = xb.Op.get_tuple_element(input_tuple, 0)
  w = xb.mkop(
      'While', [input_tuple],
      condition_computation=cond_computation,
      body_computation=body_computation)
  # w = xb.mkop(
  #     'While', [aaa_tuple.op],
  #     condition_computation=cond_computation,
  #     body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)
  hlo_print = xb.get_computation_hlo(computation)
  print("while computation: !!!!!!!!!")
  print(hlo_print)

  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 operands, computation)
  # print(met.metrics_report())

  return result
