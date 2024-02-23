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


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)


def _xla_while_loop(cond_fn, body_fn, operands):

  # def op_fn(operands):# internal_x):
  #   # TODO(manfei): replace cond_fn_placeholder and body_fn_placeholder after confirm xlacomputation could be in xla::while
  #   ## print body/cond type
  #   print("cond_fn type: ", type(cond_fn))
  #   print("body_fn type: ", type(body_fn))
  #   print("operands type: ", type(operands))

  #   ## trans body/cond to xlacomputation
  #   xm.mark_step()
  #   cond_result = cond_fn(operands)
  #   cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  #   cond_ctx_builder = cond_ctx.builder()
  #   cond_ctx_builder.name_ = 'condctx'
  #   cond_ctx.build([cond_result])
  #   cond_hlo = cond_ctx.hlo()
  #   cond_computation = xb.computation_from_module_proto("condcomputation", cond_hlo)

  #   xm.mark_step()
  #   body_result = body_fn(operands)
  #   body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  #   # body_ctx_builder = ctx.builder()
  #   # body_ctx_builder.name_ = 'bodyctx'
  #   body_ctx.build([body_result])
  #   body_hlo = body_ctx.hlo()
  #   body_computation = xb.computation_from_module_proto("bodycomputation", body_hlo)

  #   # def cond_fn_placeholder(counter, operands):
  #   #   return counter < xb.Op.scalar((operands[0]).builder(), 10, dtype=xb.Type.S32)
  #     # return counter < xb.Op.scalar((internal_x).builder(), 10, dtype=xb.Type.S32)

  #   # def body_fn_placeholder(counter, internal_x):
  #   #   next_counter = counter + xb.Op.scalar(
  #   #       counter.builder(), 1, dtype=xb.Type.S32)
  #   #   internal_x = internal_x + xb.Op.scalar(
  #   #       internal_x.builder(), 1, dtype=xb.Type.S32)
  #   #   return xb.Op.tuple((next_counter, internal_x))

  #   # zero = xb.Op.scalar(internal_x.builder(), 0, dtype=xb.Type.S32)
  #   # w = xb.Op.mkwhile((zero, internal_x), cond_fn_placeholder,
  #   #                   body_computation)

  #   ## trest operands
  #   input_tuple = Op.tuple(operands)
  #   w = input_tuple.while_loop(
  #       condition_computation=cond_computation, body_computation=body_computation)

  #   return w.get_tuple_element(1)

  # op = xor.register('test_while', op_fn)
  print("type operands: ", type(operands))
  print("operands: ", operands)
  kwargs = {}
  shapes = xb.tensor_shape(operands)
  builder = xb.create_builder('test_while')
  params = []
  secondparams = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p) # single_tuple)
    # single_tuple = xb.Op.tuple([p])
    secondparams.append(xb.Op.tuple([p]))

  # secondparams = []
  # for shape in shapes:
  #   p = xb.mkparam(builder, len(secondparams), shape)
  #   single_tuple = xb.Op.tuple([p])
  #   secondparams.append(single_tuple) # p) # single_tuple)

  xm.mark_step()
  cond_result = cond_fn(operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext() # "condctx")
  # print("type cond_ctx: ", type(cond_ctx))
  cond_ctx.setnamestring("condctx")
  # cond_builder = xb.create_builder('condctx')
  # cond_ctx_builder = cond_ctx.GetLoweringCtx().builder()
  # cond_ctx_builder.name_ = 'condctx'
  cond_ctx.build(list(cond_result))
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation", cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond_hlo: !!!!!!!!!")
  # print(cond_hlo_print)

  xm.mark_step()
  body_result = body_fn(operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext() # "bodyctx")
  body_ctx.setnamestring("bodyctx")
  # body_ctx_builder = body_ctx.builder()
  # body_ctx_builder.name_ = 'bodyctx'
  body_ctx.build(list(body_result))
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation", body_hlo)
  # body_hlo_print = xb.get_computation_hlo(body_computation)
  # print("body_hlo: !!!!!!!!!")
  # print(body_hlo_print)

  input_tuple = xb.Op.tuple(params)
  aaa_tuple = xb.Op.get_tuple_element(input_tuple, 0) # maybe move it to the cycle?
  print("aaa_tuple: ", aaa_tuple)
  print("[aaa_tuple.op]: ", [aaa_tuple.op])
  print("type [aaa_tuple.op]: ", type([aaa_tuple.op]))
  print("(aaa_tuple.op): ", (aaa_tuple.op))
  print("type (aaa_tuple.op): ", type((aaa_tuple.op)))
  w = xb.mkop('While', [aaa_tuple.op], condition_computation=cond_computation, body_computation=body_computation)
  # w = xb.mkop('While', (aaa_tuple.op,), condition_computation=cond_computation, body_computation=body_computation)
  # w = xb.mkop('While', aaa_tuple.op, condition_computation=cond_computation, body_computation=body_computation)
  # w # <torch_xla.core.xla_builder.Op object at 0x7f7d3e367f40>
  print("pass this line")
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)
  print("pass this line second @@@@@@@@@@@")

  while_loop_hlo_print = xb.get_computation_hlo(computation)
  print("while_loop_hlo: !!!!!!!!!")
  print(while_loop_hlo_print)

  # root = fn(*params, **kwargs)
  # computation = root.build(name)

  # print("operands type: ", type(operands)) # <class 'tuple'>
  # print("operands: ", operands) # (tensor([1], device='xla:0', dtype=torch.int32),)
  # print("operands[0]: ", operands[0]) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("type operands[0]: ", type(operands[0])) # <class 'torch.Tensor'>
  # print("type [operands[0],]: ", type([operands[0],])) # <class 'list'>
  # print("type (operands[0],): ", type((operands[0],))) # <class 'tuple'>
  # print("type [(operands[0],),]: ", type([(operands[0],),])) # <class 'list'>
  # localoperands = torch.tensor(1, dtype=torch.int32, device=xm.xla_device())
  localoperands = torch.tensor([1], dtype=torch.int32, device=xm.xla_device())
  # print("localoperands: ", localoperands) # tensor(1, device='xla:0', dtype=torch.int32)
  # print("999999 secondparams: ", secondparams)
  # print("999999 type secondparams: ", type(secondparams))
  # result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', secondparams, computation)
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', (localoperands,),
                                                   computation)
  # result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', [localoperands,],
  #                                                  computation)
  # _xla_user_computation:
  # [](const std::string& opname, const std::vector<at::Tensor>& inputs,
  #          const runtime::ComputationClient::ComputationPtr& computation) {
  print("done the result!!!")
  print("result: ", result)
  # op = result[0] if len(result) == 1 else result


  return result # xu.as_list(op(operands))

# ---------------------------------------
# import torch
# import torch_xla
# import torch_xla.core.xla_builder as xb
# import torch_xla.core.xla_model as xm

# device = xm.xla_device()
# a = torch.rand(1, device=device)
# b = torch.rand(1, device=device)
# c = torch.ones(1, device=device)

# name = 'fori_loop_ed_torch_func'
# opname = 'xla::_op_' + name
# kwargs = {}

# inputss = (a, b, c)
# shapess = xb.tensor_shape(inputss)

# builder = xb.create_builder(name)
# params = []
# p = xb.mkparam(builder, len(params), shapess[0]) # TODO: change to for...in
# params.append(p)
# p = xb.mkparam(builder, len(params), shapess[1])
# params.append(p)
# p = xb.mkparam(builder, len(params), shapess[2])
# params.append(p)

# def body_func(a, b, c):
#   return torch.add(a, b)

# xm.mark_step()
# result = body_func(a, b, c)
# ctx = torch_xla._XLAC.lowering.LoweringContext()
# # body_ctx_builder = ctx.builder()
# # body_ctx_builder.name_ = 'bodyctx'
# ctx.build([result])
# hlo = ctx.hlo()
# # hlo_text = ctx.hlo_text()

# def cond_func(a, b, c):
#   return c < xb.Op.scalar(c.builder(), 10, dtype=xb.Type.F32)
#   # c = c + 1
#   # return c < 10

# input_tuple = xb.Op.tuple(params) # shapess
# cond_root = cond_func(*params, **kwargs)
# cond_computation = cond_root.build(name)
# print("finish cond computation creation")

# xm.mark_step()
# cond_result = cond_func(a, b, c)
# cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
# # cond_ctx_builder = cond_ctx.builder()
# # cond_ctx_builder.name_ = 'condctx'
# cond_ctx.build([cond_result])
# cond_hlo = cond_ctx.hlo()
# cond_hlo_text = cond_ctx.hlo_text()

# body_computation = xb.computation_from_module_proto("bodycomputation", hlo)
# cond_computation = xb.computation_from_module_proto("condcomputation", cond_hlo)

# body_hlo_print = xb.get_computation_hlo(body_computation)
# cond_hlo_print = xb.get_computation_hlo(cond_computation)

# print("body_hlo: !!!!!!!!!")
# print(body_hlo_print)
# print("cond_hlo: !!!!!!!!!")
# print(cond_hlo_print)

# input_tuple = xb.Op.tuple(params) # shapess
# 1:
# w = input_tuple.while_loop(condition_computation=cond_computation, body_computation=body_computation) # w: <class 'torch_xla.core.xla_builder.Op'>
# 2:
# condition_computation = Op.make_computation('Condition', condition_computation, (input_tuple,))
# body_computation = Op.make_computation('Body', body_computation, (input_tuple,))
# w = xb.mkop('While', (input_tuple.op,), condition_computation=cond_computation, body_computation=body_computation)
# w # <torch_xla.core.xla_builder.Op object at 0x7f7d3e367f40>
# w.build(name)
