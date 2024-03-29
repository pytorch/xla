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

# _xla_get_tensor_id
  # (
  #     graph_input_tensor_ids,
  #     graph_input_xla_values,
  # ) = torch_xla._XLAC._get_tensors_xla_device_data_node(res)

# lower, upper, body_fun, init_val, one_value
# def fori_loop(upper, body_fun, lowers):#  upper, body_fun, *init_vals): # *init_val):
def fori_loop(lower, upper, body_fun, one_value, *init_val):
  # print("init_val: ", init_val)
  # val_list = list(init_val)
  # val_list.insert(0, lower)
  # val_list.insert(1, upper)

  # print("lower: ", lower) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("upper: ", upper) # tensor([20], device='xla:0', dtype=torch.int32)
  # print("body_fun: ", body_fun) # <function WhileLoopTest.test_fori_loop_tpu_addition.<locals>.body_fun at 0x7f69c40ce320>
  # print("init_val: ", init_val) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("one_value: ", one_value) # tensor([1], device='xla:0', dtype=torch.int32)

  device = xm.xla_device()

  # def cond_fn(lower, upper, x):
  def cond_fn(upper, lower, *x):
    return lower[0] < upper[0]

  # def body_fn(lower, upper, x):
  def body_fn(upper, lower, *x):
    # lower,   upper,  init_val, l_in
    # (s32[1], s32[1], s32[1],   f32[20], f32[20,10], /*index=5*/f32[10])
    one_value = torch.ones(1, dtype=torch.int32, device=device)
    # TODO(@manfei):
    # two_value = upper.clone()
    weight = torch.ones([20, 10], dtype=torch.float32, device=device)
    # return_list = list()
    # return_list.append(weight)
    return_list = list(body_fun(one_value, *x)) # s32[1], f32[20],
    # return_list.append(body_fun(one_value, *x))
    return_list.insert(2, weight) # f32[20,10]
    # weight = torch.ones([20, 10], dtype=torch.float32, device=device) # torch.linear weight
    # one_value = torch.ones(1, dtype=torch.int32, device=device)
    # return_list.append(weight)
    # return_list.append(one_value)
    return_list.insert(0, torch.sub(upper, one_value)) # s32[1]
    # return_list.insert(0, lower) # s32[1]
    # return_list.insert(-1, one_value)
    return tuple(return_list) # (torch.sub(upper, one_value), lower, body_fun(one_value, *x)) # , one_value))

  # upper, lower, one_value, init_val
  # real(ov, lower, upper, x)
  # def cond_fn(upper, lower, one_value, init_val): #one_value, lower, upper, init_val): # loop_carry): # iter, upper, one_value): # lower, *init_vals):
  def old_cond_fn(one_value, lower, upper, init_val): 
    # lower, upper, one_value, init_val = loop_carry
    lower_compare = torch.add(lower, one_value)
    # upper_compare = torch.add(upper, one_value)
    # return lower[0] <= upper[0] # while stop when cond fail
    return lower_compare[0] <= upper[0] # upper_compare[0]

  # def body_fn(upper, lowers): # , *init_vals):
  # def body_fn(upper, lower, one_value, init_val): # one_value, lower, upper, init_val): # loop_carry): # iter, upper, one_value):
  def old_body_fn(one_value, lower, upper, init_val):
    # lower, upper, one_value, init_val = loop_carry
    # return (torch.add(iter, one_value).clone(), upper.clone(), one_value.clone(), body_fun(x, one_value).clone())
    # one_value = torch.tensor([1], dtype=torch.int32, device=device)
    # new_upper = torch.sub(upper, one_value)
    new_lower = torch.add(lower, one_value)
    new_init_val = body_fun(init_val, one_value)
    # return (new_lower, upper, one_value, new_init_val)
    # return (new_upper, lower, one_value, new_init_val) # one_value, lower, new_upper, new_init_val)
    # return (upper, new_lower, one_value, new_init_val)
    return (one_value, new_lower, upper, new_init_val)

  # loop_carruy_print = (lower, upper, one_value, init_val)
  # print("loop_carruy_print[0]: ", loop_carruy_print[0]) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("loop_carruy_print[1]: ", loop_carruy_print[1]) # tensor([20], device='xla:0', dtype=torch.int32)
  # print("loop_carruy_print[2]: ", loop_carruy_print[2]) # tensor([1], device='xla:0', dtype=torch.int32)
  # print("loop_carruy_print[3]: ", loop_carruy_print[3]) # tensor([1], device='xla:0', dtype=torch.int32)

  # res = _xla_while_loop(cond_fn, body_fn, upper, lower, one_value, init_val) # one_value, lower, upper, init_val) # upper, lower, one_value, init_val)
  # res = _xla_while_loop(cond_fn, body_fn, one_value, lower, upper, init_val)
  # print("init_val: ", init_val)
  # print("init_val[0]: ", init_val[0])
  # print("init_val[1]: ", init_val[1])
  # print("type init_val[0]: ", type(init_val[0]))
  # print("type init_val[1]: ", type(init_val[1]))
  # print("init_val[2]: ", init_val[2])
  # print("init_val[3]: ", init_val[3])

  if len(init_val) >= 1:
    val_list = list(init_val)
    val_list.insert(0, lower)
    val_list.insert(1, upper)
    # print("val_list: ", val_list)
    res = _xla_while_loop(cond_fn, body_fn, tuple(val_list))
    # upper, lower, body_fun, one_value, init_val, l_in
    # lower, upper, init_val, l_in
    return res
  else:
    # TODO(@manfei): this should not arrived, due to init_val must contain value
    res = _xla_while_loop(cond_fn, body_fn, lower, upper, *init_val)
    return res

@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)


# fori_loop: original_operands==(lower, upper, init_val)
# def _xla_while_loop(cond_fn, body_fn, original_operands):
# (lower, upper, one_value, init_val)
def _xla_while_loop(cond_fn, body_fn, *operands):
  # print("!!! arguments: original_operands: ", original_operands)
  # fake operands to split formal code
  # operands = [] # fake_operands
  # for original_operand in original_operands:
  #   device = original_operand.device
  #   operands.append(torch.randint(10, original_operand.size(), dtype=torch.int32).to(device))
  # operands = tuple(operands)
  # print("!!! operands: ", operands) # (tensor([0], device='xla:0', dtype=torch.int32), tensor([30], device='xla:0', dtype=torch.int32), tensor([1], device='xla:0', dtype=torch.int32))

  # print("!!! arguments: cond_fn: ", cond_fn, ", body_fn: ", body_fn, ", operands: ", operands)
  # cond_fn: <function fori_loop.<locals>.cond_fn at 0x7f469149e710>
  # body_fn: <function fori_loop.<locals>.body_fn at 0x7f469149e680>
  # operands: (tensor([1], device='xla:0', dtype=torch.int32),
  #            tensor([20], device='xla:0', dtype=torch.int32),
  #            tensor([1], device='xla:0', dtype=torch.int32),
  #            tensor([1], device='xla:0', dtype=torch.int32))

  # create inputs placeholder
  # operands_tuple = tuple(operands)
  # print("in _xla_while_loop: ", operands)
  kwargs = {}
  # extend input operands to include complex model generated more arguments
  # cond_fn, body_fn, lower, upper, *init_val = operands
  # placeholder_func = torch.rand(size = l_out.size(), device = device)
  # placeholder_input = torch.rand(size = l_in_i.size(), device = device)

  if type(operands) is tuple:
    # print("aaa")
    operands = operands[0]
    shapes = xb.tensor_shape(operands)
  else:
    # print("bbb")
    shapes = xb.tensor_shape((operands)) # _tuple)
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # lower, upper, init_val = operands
  # lower, upper, one_value, init_val = operands
  # print("arrive here!!!")
  # generate cond_fn xlacomputation
  # print("!!! operands: ", operands)
  # !!! operands:  
  #     (tensor([1], device='xla:0', dtype=torch.int32),
  #     tensor([20], device='xla:0', dtype=torch.int32),
  #     tensor([1], device='xla:0', dtype=torch.int32),
  #     tensor([1], device='xla:0', dtype=torch.int32))

  # generate body_fn xlacomputation
  body_result = body_fn(*operands) # lower, upper, init_val) # operands) # *operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  # body_ctx.build(list(body_result))
  body_ctx.build(list(body_result), []) # [one_value, init_val]) # , [init_val])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # import pdb; pdb.set_trace()
  # analyze body_hlo_print, get body_xlacomputation's input/output * check same
  body_hlo_print_first_line = (body_hlo_print.split(")}", 1))[0]
  print("body_hlo_print_first_line: ", body_hlo_print_first_line)
  entry_computation_layout = (body_hlo_print_first_line.split(", entry_computation_layout={"))[1][2:]
  inputs_shape, outputs_shape = entry_computation_layout.split("->", 1)
  if inputs_shape[:-2] != outputs_shape[1:]:
    print("[ERROR]: body_xlacomputation's input and output are not the same!!!")
  # outputs_shape = (s32[1]{0}, s32[1]{0}, s32[1]{0}, f32[20]{0}, f32[20]{0}, /*index=5*/f32[10]{0})
  # filter all item in outputs_shape and trans to `cond_ctx.build` to add new params when build cond xlacomputation
  outputs_shape_list = outputs_shape[1:-1].split(", ")
  print("outputs_shape_list: ", outputs_shape_list)
  additional_arguments = []
  for i in outputs_shape_list[2:]: # skip upper and lower
    if (i[:2]=='/*'): # clean prefix like /*...*/
      i = (i.split('*/'))[1]

    if (i[-3:]=='{0}'): # check end with {0}
      # if (i[:3]=='s32'): # xla::PrimitiveType::S32
      i_size_number = int(i[4:-4])
      additional_arguments.append((i[:3], i_size_number))
    # additional_arguments.append(i[])
    else: # not end with {0}
      additional_arguments.append(('s64', 0)) # s32[1] # xla::PrimitiveType::S32

  # create tensors based on additional_arguments
  additional_tensors = []
  device = operands[0].device
  for i, j in additional_arguments:
    # [('s32', 1), ('f32', 20), ('f32', 20), ('f32', 10), ('s64', 0)]
    if i=='s32':
      additional_tensors.append(torch.ones(j, dtype=torch.int32, device=device))
    elif i=='f32':
      additional_tensors.append(torch.ones(j, dtype=torch.float32, device=device))
    elif i=='s64':
      additional_tensors.append(torch.ones(j, dtype=torch.int64, device=device))
    else:
      additional_tensors.append(torch.ones(j, dtype=torch.int32, device=device))

  cond_result = cond_fn(*operands) # lower, upper, init_val) # operands) # *operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  # print("arrive here!!!")
  # print("cond_result: ", cond_result)
  # print("init_val: ", init_val)
  # TODO(@manfei) to reduce to operands[2:]
  cond_ctx.build([cond_result], additional_tensors) # list(operands[2:]))# operands[:1], operands[3:])) # [one_value, init_val]) # , init_val) # [operands[2]])
  # print("arrive here!!!")
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  print("body: operands: ", operands)

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
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while', (operands), computation)

  # print("operands: ", operands)
  # print("upper: ", operands[0])
  # print("lower: ", operands[1])
  # print("init: ", operands[2])
  # print("in _xla_while_loop result: ", result)
  return result


##################### old tried code
 # limit_value = upper
  # init = lower
  # iterator = lower

  # one_value is actually not used here, but actually redefined in body_fn to avoid introduce new argument in body_xlacomputation
  # lower == init_val
  # assert(lower == init_val)
  # init = lower # = init_val
  # limit_value = upper

  # one_value_original = torch.tensor([1], dtype=torch.int32, device=device)
  # (a, b) = init_vals

    # def cond_fn(init, limit_value):
    #   return limit_value[0] >= init[0]

    # def body_fn(init, limit_value):
    #   one_value = torch.ones(1, dtype=torch.int32, device=device)
    #   return (torch.add(init, one_value), limit_value.clone())

  # cond_fn
  # def _fori_cond_fun(loop_carry):
  #   i, upper, _ = loop_carry
  #   return torch.lt(i, upper)

  # def _fori_body_fun(body_fun):
  #   # body_fun = weakref.ref(body_fun)
  #   def while_body_fun(loop_carry):
  #     i, upper, x = loop_carry
  #     one_value = torch.ones(1, dtype=torch.int32, device=device)
  #     # return torch.add(i, one_value), upper, body_fun(i, x) # body_fun()(i, x)
  #     return torch.add(i, one_value), upper, body_fun(x) # body_fun()(i, x)
  #   return while_body_fun

  # # def cond_fn(upper, lowers): # lower, *init_vals):
  # def cond_fn(init, limit_value): # lower, *init_vals):
  #   # init_val_compy = init_val.clone()
  #   # one_value1 = torch.tensor([0], dtype=torch.int32, device=device)
  #   # one_value2 = torch.tensor([0], dtype=torch.int32, device=device)
  #   # lower = torch.add(lower, one_value1[0])
  #   # lower = torch.sub(lower, one_value2[0])
  #   # assert isinstance(init_vals[0], torch.Tensor)
  #   # assert isinstance(init_vals[1], torch.Tensor)
  #   # bool_value = isinstance(init_vals[0], torch.Tensor) and isinstance(init_vals[1], torch.Tensor)
  #   # body_fun(*init_vals)
  #   # result = True
  #   # if (lower[0] <= upper[0]) and bool_value:
  #   #   return True
  #   # return False
  #   # bool_result = ((lower[0] <= upper[0]) and bool_value)
  #   # bool_tensor = torch.tensor(bool_result, dtype=torch.bool)
  #   # return bool_tensor # (lower[0] <= upper[0]) and bool_tensor
  #   # return lower[0] <= upper[0]
  #   # return lowers[0] <= upper[0]
  #   return limit_value[0] >= init[0]


  #   # one_value_original = torch.tensor(1, dtype=torch.int32, device=device)
  #   # (a, b) = init_vals
  #   # return (upper, torch.add(lower, 1), body_fun(a, b), b.clone())
  #   # return (upper.clone(), (torch.add(lower.clone(), init_vals[1].clone())).clone(), (body_fun(*init_vals)).clone(), init_vals[1].clone()) # init_vals[1:])
  #   # return (upper, (torch.add(lowers[0], lowers[2]), body_fun(lowers[1], lowers[2]), lowers[2])) # init_vals[1:])
  #   # (body_fun(*init_vals)).clone(), init_vals[1].clone())
  #   # body_fun(one_value_original, init_val)) # body_fun(lower, init_val))
  #   one_value = torch.ones(1, dtype=torch.int32, device=device)


  # res = while_loop(cond_fn, body_fn, (upper, lower, *init_vals))
  # lowers = (lower, *init_vals)
  # res = _xla_while_loop(cond_fn, body_fn, (upper, lowers)) # , *init_vals))
  # lower, upper, body_fun, init_val, one_value
  # res = _xla_while_loop(cond_fn, body_fn, (init, limit_value))
  # res = _xla_while_loop(cond_fn, body_fn, (lower, upper, init_val))


  # inits): # init_val, one_value):
  # _, _, result = _xla_while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
  #                           (lower, upper, inits))
  # _, _, result = _xla_while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
  #                           (lower, upper, init_val))
  # print("upper: ", upper)
  # print("lower: ", lower)