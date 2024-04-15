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


# TODO(@manfei): delete one_value?
def fori_loop(upper, lower, body_fun, init_val, *input_value):

  device = xm.xla_device()

  def cond_fn(upper, lower, one_value, x, *input_value, weight_0, output_value, bias_0): # , bias_0): # , output_value):
    return lower[0] < upper[0]

  def body_fn(upper, lower, one_value, x, *input_value, weight_0, output_value, bias_0): # , bias_0):
    # weight = body_fun.weight
    new_lower = torch.add(one_value, lower) ### !!! this matter, torch.add might would change the second argument's value, even we use a new variable to catch the result!!!
    output_value = body_fun(*input_value) ### !!! due to the output_value is not actually used here, 
    # --- !!! its original value would not be used, and it would be replaces by the result of body_fun
    # --- !!! so, due to PTXLA is traced from result tensor, so the arguments `output_value` would not be included in the body_xlacomputation
    # --- !!! so, we need to modify ini_python_binding.cpp to add a fake arguments in the xlacompputation
    weight = body_fun.weight
    bias = body_fun.bias
    return upper, new_lower, one_value, torch.add(one_value, x), *input_value, weight, bias, output_value

  output_value = torch.zeros([20], dtype=torch.float32, device=device)
  weight_0 = body_fun.weight
  bias_0 = body_fun.bias
  one_value = torch.tensor([1], dtype=torch.int32, device=device)
  res = while_loop(cond_fn, body_fn, (upper, lower, one_value, init_val, *input_value, weight_0, bias_0, output_value))
  return res


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs):
  # TODO(@manfei): PyTorch require carried_inputs to be list/tuple, PyTorch/XLA _xla_while_loop only accept *operands, *operands would tuple items again: (a, '')
  # cond_fn&body_fn: callable
  # carried_inputs: (Tuple of possibly nested dict/list/tuple of tensors)
  print("!!! arrive here too !!!")
  print("while_loop additional_inputs: ", additional_inputs)
  if additional_inputs is None:
    additional_inputs = tuple()
  return _xla_while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs=additional_inputs) #  a=a, b=b, c=c,


def _xla_while_loop(cond_fn, body_fn, *original_carried_inputs, additional_inputs=()):
  print("!!! arrive here too too !!!")
  # print("carried_inputs: ", carried_inputs)
  print("additional_inputs: ", additional_inputs)
  # import pdb; pdb.set_trace()
  # untuple carried_inputs from while_loop
  carried_inputs = original_carried_inputs[0]
  # TODO(@manfei): please clear pass additional_inputs in `while_loop`'s defination in this file
  if len(original_carried_inputs) == 2:
    print("use original_carried_inputs for additional_inputs")
    additional_inputs = original_carried_inputs[1]
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    #TODO(@manfei) type = carried_input.type
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  # fake_carried_inputs = tuple(fake_carried_inputs)
  # print("fake_carried_inputs first: ", fake_carried_inputs)
  # for additional_input in additional_inputs:
  #   device = additional_input.device
  #   #TODO(@manfei) type = carried_input.type
  #   fake_carried_inputs.append(
  #       torch.randint(10, additional_input.size(),
  #                     dtype=additional_input.dtype).to(device))
  # fake_carried_inputs = tuple(fake_carried_inputs)
  # # print("fake_carried_inputs second: ", fake_carried_inputs)

  print("!!! arrive here too before cond !!!")
  # generate cond_fn xlacomputation
  print("print fake_carried_inputs: ", fake_carried_inputs)
  # TODO(@manfei): specify which element is for which argument like a,b,c
  cond_result = cond_fn(*fake_carried_inputs) # [:-3], weight_0=fake_carried_inputs[-2], output_value=fake_carried_inputs[-3], bias_0=fake_carried_inputs[-1])
  print("nnn here ???")
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  additional_inputs_list_cond = list(fake_carried_inputs[2:]) # all missed arguments except upper/lower due to PyTorch/XLA trace from output tensor
  # treat and pass additional_inputs to cond_fn
  print("additional_inputs_list_cond one: ", additional_inputs_list_cond)
  for i in range(len(additional_inputs)):
    additional_inputs_list_cond.append(additional_inputs[i])
  print("additional_inputs_list_cond two: ", additional_inputs_list_cond)
  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  print("!!! arrive here too after cond !!!")

  print("!!! arrive here too before body !!!")
  # generate body_fn xlacomputation
  body_result = body_fn(*fake_carried_inputs) # [:-3], weight_0=fake_carried_inputs[-1], output_value=fake_carried_inputs[-3], bias_0=fake_carried_inputs[-2])
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  additional_inputs_list_body = [fake_carried_inputs[-2]] # missed arguments due to given output_value was not used and PyTorch/XLA trace xlacomputation from output tensor
  # TODO(@manfei): treat and pass additional_inputs to body_fn too
  # print("list(fake_carried_inputs[-2]: ", fake_carried_inputs[-2])
  # print("len0!!!: ", len(additional_inputs_list_body))
  for i in range(len(additional_inputs)):
    additional_inputs_list_body.append(additional_inputs[i])
  # print("len!!!: ", len(additional_inputs_list_body))
  # print("additional_inputs_list_body: ", additional_inputs_list_body)
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  print("!!! arrive here too after body !!!")

  print("!!! arrive here too before args!!!")
  # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
  kwargs = {}
  if type(carried_inputs) is tuple:
    shapes = xb.tensor_shape(carried_inputs)
  else:
    shapes = xb.tensor_shape((carried_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)
  print("!!! arrive here too after args!!!")

  print("!!! arrive here too before while!!!")
  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (carried_inputs),
                                                 computation)
  print("!!! arrive here too after while!!!")

  return result