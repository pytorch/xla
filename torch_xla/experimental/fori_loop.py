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
from torch._higher_order_ops.while_loop import while_loop as torch_while_loop


def insert_model_pars_into_additional_inputs(weight_bias_lists, layer_named_parameters):
  for name, param in layer_named_parameters:
    # print("name: ", name)
    # print("param: ", param)
    weight_bias_lists.append(param)

# # TODO(@manfei): treat *input_value
# def fori_loop(upper, lower, body_fun, init_val, input_value):

#   device = xm.xla_device()

#   one_value = torch.tensor([1], dtype=torch.int32, device=device)

#   if (hasattr(body_fun, 'weight') or hasattr(body_fun, 'bias')):
#     output_value = torch.zeros([20], dtype=torch.float32, device=device)

#     def cond_fn(upper, lower, one_value, x, input_value, output_value):
#       return lower[0] < upper[0]

#     def body_fn(upper, lower, one_value, x, input_value, output_value):
#       new_lower = torch.add(one_value, lower)
#       output_value = body_fun(input_value)
#       weight = body_fun.weight  # not be used actually, initialized as placeholder xlacomputation requirement
#       bias = body_fun.bias  # not be used actually, initialized as placeholder xlacomputation requirement
#       return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
#           one_value, x), input_value.clone(), bias.clone(), weight.clone(
#           ), output_value.clone()

#     res = torch_while_loop(
#         cond_fn, body_fn,
#         (upper, lower, one_value, init_val, input_value, output_value))
#   else:
#     output_value = torch.tensor([1], dtype=torch.int32, device=device)

#     def cond_fn(upper, lower, one_value, x, input_value):
#       return lower[0] < upper[0]

#     def body_fn(upper, lower, one_value, x, input_value):
#       new_lower = torch.add(one_value, lower)
#       output_val = body_fun(one_value, input_value)
#       return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
#           one_value, x), output_val.clone()

#     res = torch_while_loop(cond_fn, body_fn,
#                            (upper, lower, one_value, init_val, input_value))

#   return res


# @while_loop_op.py_impl(DispatchKey.XLA)
# def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=None):
#   # TODO(@manfei): PyTorch require carried_inputs to be list/tuple, PyTorch/XLA _xla_while_loop only accept *operands, *operands would tuple items again: (a, '')
#   # cond_fn&body_fn: callable
#   # carried_inputs: (Tuple of possibly nested dict/list/tuple of tensors)
#   if additional_inputs is None:
#     additional_inputs = tuple()
#   # print("arrive @while_loop_op.py_impl(DispatchKey.XLA)")
#   return _xla_while_loop(cond_fn, body_fn, carried_inputs, additional_inputs)
#   # return _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs)


# # ----------------- PyLoweringContext --------------------------------
# # MNIST's _xla_while_loop's pre func to summary args
# def _xla_while_loop_target_first(cond_fn, body_fn, carried_inputs, additional_inputs=None):
#   def new_body_fn(*carried_inputs):
#     res = list(body_fn(*carried_inputs))

#     # iter = res[0]
#     # inputs_and_outputs = res[1:]
#     # # if len(inputs_and_outputs)==1:
#     # #   inputs_and_outputs = [inputs_and_outputs,]
#     # # res = res + list(additional_inputs)
#     # res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
#     # return res

#     iter_inputs = res[:-1]
#     outputs = res[-1]
#     # if len(inputs_and_outputs)==1:
#     #   inputs_and_outputs = [inputs_and_outputs,]
#     # res = res + list(additional_inputs)
#     # res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
#     res = iter_inputs + list(outputs, )
#     return res
#   return _xla_while_loop_target(cond_fn, new_body_fn, carried_inputs, additional_inputs)

# # MNIST's _xla_while_loop with PyLoweringContext
# def _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs=None):

#   output_value_index = len(carried_inputs) - 1
#   print("current output_value_index: ", output_value_index)
#   # print("carried_inputs: ", carried_inputs)
#   print("--- --- --- carried_inputs --- --- ---")
#   for i in range(len(carried_inputs)):
#     print("carried_inputs ", i, " size: ", carried_inputs[i].size())
#   print("--- --- --- additional_inputs --- --- ---")
#   # print("additional_inputs: ", additional_inputs)
#   for i in range(len(additional_inputs)):
#     print("additional_inputs ", i, " size: ", additional_inputs[i].size())

#   fake_carried_inputs = []
#   for carried_input in carried_inputs:
#     device = carried_input.device
#     fake_carried_inputs.append(
#         torch.randint(10, carried_input.size(),
#                       dtype=carried_input.dtype).to(device))
#   for additional_input in additional_inputs:
#     device = additional_input.device
#     fake_carried_inputs.append(
#         torch.randint(
#             10, additional_input.size(),
#             dtype=additional_input.dtype).to(device))

#   # print("fake_carried_inputs: ", fake_carried_inputs.size)
#   print("--- --- --- fake_carried_inputs --- --- ---")
#   for i in range(len(fake_carried_inputs)):
#     print("fake_carried_inputs ", i, " size: ", fake_carried_inputs[i].size())

#   # TODO(@manfei): specify which element is for which argument like a,b,c
#   cond_result = cond_fn(*fake_carried_inputs)
#   cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
#   cond_ctx.set_name_string("condctx")

#   # TODO(@manfei): treat hard-code cond xlacomputation change: currently switch output_value and weight position if additional_inputs(weight/bias) exists
#   additional_inputs_list_cond = list(
#       # fake_carried_inputs[2:]
#       fake_carried_inputs[2:]
#   )  # all missed arguments except upper/lower due to PyTorch/XLA trace from output tensor
#   # reorder the additional_inputs due to the given additional_inputs are not generated with expected order, let's check how `additional_inputs` was generated for mnist
#   if additional_inputs:
#     # print("arrive here for cond !!!")
#     tmp_output = additional_inputs_list_cond[0] # len(carried_inputs) - 1]# 3]  # not used, change order doesn't affect logic
#     del additional_inputs_list_cond[0] # len(carried_inputs) - 1] # 3]  # not used, change order doesn't affect logic
#     additional_inputs_list_cond.append(tmp_output)  # not used, change order doesn't affect logic

#   cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
#   # cond_ctx.buildforiloop([cond_result], ())
#   cond_hlo = cond_ctx.hlo()
#   cond_computation = xb.computation_from_module_proto("condcomputation",
#                                                       cond_hlo)
#   cond_hlo_print = xb.get_computation_hlo(cond_computation)
#   print("cond computation: !!!!!!!!!")
#   print(cond_hlo_print)

#   # import pdb; pdb.set_trace()
#   # print("additional_inputs: ", additional_inputs)
#   # generate body_fn xlacomputation
#   body_result = body_fn(*fake_carried_inputs)
#   body_ctx = torch_xla._XLAC.lowering.LoweringContext()
#   body_ctx.set_name_string("bodyctx")

#   # TODO(@manfei): treat hard-code body xlacomputation change: currently add non-changed output_value argument if additional_inputs(weight/bias) exists
#   if additional_inputs:
#     # print("arrive here !!!")
#     additional_inputs_list_body = [fake_carried_inputs[(len(carried_inputs) - 1)]]
#   else:
#     # print("arrive here too !!!")
#     additional_inputs_list_body = []

#   # TODO(@manfei): treat hard-code parameters: additional_inputs_list_body
#   body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
#   # body_ctx.buildforiloop(list(body_result), ())
#   body_hlo = body_ctx.hlo()
#   body_computation = xb.computation_from_module_proto("bodycomputation",
#                                                       body_hlo)
#   # body_hlo_print = xb.get_computation_hlo(body_computation)
#   # print("body computation: !!!!!!!!!")
#   # print(body_hlo_print)

#   # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
#   total_inputs = carried_inputs + tuple(additional_inputs)
#   kwargs = {}
#   if type(total_inputs) is tuple:
#     shapes = xb.tensor_shape(total_inputs)
#   else:
#     shapes = xb.tensor_shape((total_inputs))
#   builder = xb.create_builder('test_while')
#   params = []
#   for shape in shapes:
#     p = xb.mkparam(builder, len(params), shape)
#     params.append(p)

#   # TODO(@manfei): treat hard-code input arguments, currently switch bias and output_value if additional_inputs(weight/bias) exists
#   if additional_inputs:
#     tmp_output = params[(len(carried_inputs) - 1)]# 5]
#     del params[(len(carried_inputs) - 1)]# 5]
#     params.append(tmp_output)
#     # tmp_bias = params[-3]
#     # del params[-3]
#     # params.append(tmp_bias)

#   # generate while xlacomputation
#   input_tuple = xb.Op.tuple(tuple(params))
#   w = xb.mkop(
#       'While', (input_tuple.op,),
#       condition_computation=cond_computation,
#       body_computation=body_computation)
#   name = 'fori_loop_ed_torch_func'
#   computation = w.build(name)

#   # gain final result with generated while xlacomputation
#   result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
#                                                  (total_inputs), computation)

#   return result





# ------------------ context ---------------------------------------------------------------------
# import numpy as np
# import torch
# import torch_xla
# from torch._ops import HigherOrderOperator
# import torch._higher_order_ops.while_loop
# from torch._higher_order_ops.while_loop import while_loop_op
# from torch._higher_order_ops.while_loop import while_loop as torch_while_loop

# TODO(@manfei): treat *input_value
def fori_loop(upper, lower, body_fun, init_val, input_value):

  device = xm.xla_device()

  one_value = torch.tensor([1], dtype=torch.int32, device=device)

  if (hasattr(body_fun, 'weight') or hasattr(body_fun, 'bias')):
    output_value = torch.zeros([20], dtype=torch.float32, device=device)

    def cond_fn(upper, lower, one_value, x, input_value, output_value):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value, output_value):
      new_lower = torch.add(one_value, lower)
      output_value = body_fun(input_value)
      weight = body_fun.weight  # not be used actually, initialized as placeholder xlacomputation requirement
      bias = body_fun.bias  # not be used actually, initialized as placeholder xlacomputation requirement
      return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
          one_value, x), input_value.clone(), bias.clone(), weight.clone(
          ), output_value.clone()

    res = torch_while_loop(
        cond_fn, body_fn,
        (upper, lower, one_value, init_val, input_value, output_value))
  else:
    output_value = torch.tensor([1], dtype=torch.int32, device=device)

    def cond_fn(upper, lower, one_value, x, input_value):
      return lower[0] < upper[0]

    def body_fn(upper, lower, one_value, x, input_value):
      new_lower = torch.add(one_value, lower)
      output_val = body_fun(one_value, input_value)
      return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
          one_value, x), output_val.clone()

    res = torch_while_loop(cond_fn, body_fn,
                           (upper, lower, one_value, init_val, input_value))

  return res

@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  # TODO(@manfei): PyTorch require carried_inputs to be list/tuple, PyTorch/XLA _xla_while_loop only accept *operands, *operands would tuple items again: (a, '')
  # cond_fn&body_fn: callable
  # carried_inputs: (Tuple of possibly nested dict/list/tuple of tensors)
  if additional_inputs is None:
    additional_inputs = tuple()
  # print("arrive @while_loop_op.py_impl(DispatchKey.XLA)")
  # return _xla_while_loop(cond_fn, body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target_first_second(cond_fn, body_fn, carried_inputs, additional_inputs)
  print("dispatchkey here !!!")
  # print("additional_inputs size: ", additional_inputs.size())
  # print("additional_inputs: ", additional_inputs)
  # for i in range(len(additional_inputs)): print("additional_inputs: ", i, " size: ", additional_inputs.size()) # ()
  # return _xla_while_loop_target_first_second_clean_version_s32_may17_1018am(cond_fn, body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32_may17_1528pm(cond_fn, body_fn, carried_inputs, additional_inputs)
  return _xla_while_loop_target_second_clean_version_s32_may19_2205pm(cond_fn, body_fn, carried_inputs, additional_inputs)

def _xla_while_loop_target_first(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  def new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
    return res

  def try_new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    # res = body_fn(*carried_inputs)
    # iter = res[0]
    # inputs_and_outputs = res[1:]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    res = res + list(additional_inputs)
    # res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
    return res

  def second_try_new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    # res = body_fn(*carried_inputs)
    # iter = res[0]
    # inputs_and_outputs = res[1:]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    res = list(additional_inputs) + res
    # res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
    return res

  def third_try_new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    res = [iter,] + inputs_items + additional_inputs + [outputs_items, ]
    return res

  def forth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, one] + inputs_items + additional_inputs + [outputs_items, ]
    return res

  def fifth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    # res = iter, 1, additiona_inputs, inputs, outputs
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, one] + additional_inputs + inputs_items + [outputs_items, ]
    return res

  # # xla_device = carried_inputs[0].device
  # xla_device = additional_inputs[0].device
  # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
  # additional_inputs = [one, ] + additional_inputs

  def sixth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    # res = iter, 1, additiona_inputs, inputs, outputs
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs) + inputs_items + [outputs_items, ]
    return res

  def seventh_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, one] + additional_inputs + inputs_items + [outputs_items, ]
    return res

  def eigth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + additional_inputs + [one, ] + inputs_items + [outputs_items, ]
    return res

  def ninth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs[0]) + [one, ] + list(additional_inputs[1]) + inputs_items + [outputs_items, ]
    return res

  # new_additional_inputs = additional_inputs[0] + additional_inputs[1]

  # return _xla_while_loop_target(cond_fn, new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, second_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, third_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, forth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, fifth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, sixth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, seventh_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, eigth_try_new_body_fn, carried_inputs, additional_inputs)
  return _xla_while_loop_target(cond_fn, ninth_try_new_body_fn, carried_inputs, additional_inputs)

# used new s32
def _xla_while_loop_target_second_clean_version_s32_may19_2205pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  def new_body_fn(iter_inputs_values, one, *ouputs_values): # *carried_inputs):
    res = list(body_fn(iter_inputs_values, *ouputs_values))

    res_iter_inputs = res[:-1]
    res_outputs = res[-1]

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [one, ] + [res_outputs, ]
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      res = list(res_iter_inputs) + list(additional_inputs) + [one, ] + [res_outputs, ]
    else: # no additional_inputs and no bn_additional_inputs
      res = list(res_iter_inputs) + [one, ] + [res_outputs, ]
    return res

  def new_body_fn_may16_1426pm(*carried_inputs):
    res = list(body_fn(*carried_inputs))

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [res_outputs, ]
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      res = list(res_iter_inputs) + list(additional_inputs) + [res_outputs, ]
    else: # no additional_inputs and no bn_additional_inputs
      res = res # list(res_iter_inputs) + [res_outputs, ]
    return res

  print("wrapper carried_inputs: ", carried_inputs)
  def new_body_fn_may19_2208pm(*carried_inputs):
    res = list(body_fn(*carried_inputs))

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [res_outputs, ]
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      # res = list(res_iter_inputs) + list(additional_inputs) + [res_outputs, ]
      # res = [res[0], ] + list(additional_inputs) + [res[1:], ]
      res = [res[0], ] + list(additional_inputs) + res[1:]
    else: # no additional_inputs and no bn_additional_inputs
      res = res # list(res_iter_inputs) + [res_outputs, ]
    return res

  return _xla_while_loop_target_second_clean_version_s32_may19_2206pm(cond_fn, new_body_fn_may19_2208pm, carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_may17_1018am(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  def new_body_fn(iter_inputs_values, one, *ouputs_values): # *carried_inputs):
    res = list(body_fn(iter_inputs_values, *ouputs_values))

    res_iter_inputs = res[:-1]
    res_outputs = res[-1]

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [one, ] + [res_outputs, ]
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      res = list(res_iter_inputs) + list(additional_inputs) + [one, ] + [res_outputs, ]
    else: # no additional_inputs and no bn_additional_inputs
      res = list(res_iter_inputs) + [one, ] + [res_outputs, ]
    return res

  def new_body_fn_may16_1426pm(*carried_inputs):
    res = list(body_fn(*carried_inputs))

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [res_outputs, ]
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      res = list(res_iter_inputs) + list(additional_inputs) + [res_outputs, ]
    else: # no additional_inputs and no bn_additional_inputs
      res = res # list(res_iter_inputs) + [res_outputs, ]
    return res

  return _xla_while_loop_target_second_clean_version_s32_may17_1018am(cond_fn, body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_may16_2137pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  # xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device)

  # iter = carried_inputs[0]
  # inputs_ouputs_val = carried_inputs[1:]
  # new_carried_inputs = [iter, one] + list(inputs_ouputs_val)

  # iter = carried_inputs[0]
  # inputs_values = carried_inputs[1:-1]
  # iter_inputs_values = carried_inputs[:-1]
  # # TODO(@manfei): get length of ouputs_values
  # ouputs_values = carried_inputs[-1]
  # # new_carried_inputs = [iter, one] + [inputs_values, ] + list([ouputs_values, ])
  # new_carried_inputs = list(iter_inputs_values) + [one, ] + list([ouputs_values, ])

  # # def new_cond_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
  # def new_cond_fn(iter_inputs_values, _, *ouputs_values): # *carried_inputs):
  #   return cond_fn(iter_inputs_values, *ouputs_values)

  def new_body_fn(iter_inputs_values, one, *ouputs_values): # *carried_inputs):
    res = list(body_fn(iter_inputs_values, *ouputs_values))

    res_iter_inputs = res[:-1]
    res_outputs = res[-1]

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [one, ] + [res_outputs, ]
      # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      # res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      res = list(res_iter_inputs) + list(additional_inputs) + [one, ] + [res_outputs, ]
      # res.insert(1, one)
      # res.insert(1, additional_inputs)
      # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    else: # no additional_inputs and no bn_additional_inputs
      # print("arrive here 3 !!!")
      # print("res: ", res)
      # print("arrive here 3-1 !!!")
      # res = [iter, ] + [one, ] + inputs_and_outputs
      res = list(res_iter_inputs) + [one, ] + [res_outputs, ]
      # res = res.insert(1, one) # body_result would be None
    # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    return res

  def new_body_fn_may16_1426pm(*carried_inputs):
    res = list(body_fn(*carried_inputs))

    # res_iter_inputs = res[:-1]
    # res_outputs = res[-1]

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
      # res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [one, ] + [res_outputs, ]
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [res_outputs, ]
      # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      # res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      # res = list(res_iter_inputs) + list(additional_inputs) + [one, ] + [res_outputs, ]
      res = list(res_iter_inputs) + list(additional_inputs) + [res_outputs, ]
      # res.insert(1, one)
      # res.insert(1, additional_inputs)
      # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    else: # no additional_inputs and no bn_additional_inputs
      # print("arrive here 3 !!!")
      # print("res: ", res)
      # print("arrive here 3-1 !!!")
      # res = [iter, ] + [one, ] + inputs_and_outputs
      # res = list(res_iter_inputs) + [one, ] + [res_outputs, ]
      res = res # list(res_iter_inputs) + [res_outputs, ]
      # res = res.insert(1, one) # body_result would be None
    # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    return res

  # def new_new_cond_fn(*carried_inputs):
  #   return cond_fn(*carried_inputs)

  # def new_new_body_fn(*carried_inputs):
  #   return body_fn(*carried_inputs)

  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32_may16_1617pm(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32_may16_1617pm(cond_fn, new_body_fn_may16_1426pm, carried_inputs, additional_inputs, bn_additional_inputs)
  return _xla_while_loop_target_second_clean_version_s32_may16_2138pm(cond_fn, body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_new_cond_fn, new_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_may16_1603pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  # xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device)

  # iter = carried_inputs[0]
  # inputs_ouputs_val = carried_inputs[1:]
  # new_carried_inputs = [iter, one] + list(inputs_ouputs_val)

  # iter = carried_inputs[0]
  # inputs_values = carried_inputs[1:-1]
  # iter_inputs_values = carried_inputs[:-1]
  # # TODO(@manfei): get length of ouputs_values
  # ouputs_values = carried_inputs[-1]
  # # new_carried_inputs = [iter, one] + [inputs_values, ] + list([ouputs_values, ])
  # new_carried_inputs = list(iter_inputs_values) + [one, ] + list([ouputs_values, ])

  # # def new_cond_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
  # def new_cond_fn(iter_inputs_values, _, *ouputs_values): # *carried_inputs):
  #   return cond_fn(iter_inputs_values, *ouputs_values)

  def new_body_fn(iter_inputs_values, one, *ouputs_values): # *carried_inputs):
    res = list(body_fn(iter_inputs_values, *ouputs_values))

    res_iter_inputs = res[:-1]
    res_outputs = res[-1]

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [one, ] + [res_outputs, ]
      # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      # res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      res = list(res_iter_inputs) + list(additional_inputs) + [one, ] + [res_outputs, ]
      # res.insert(1, one)
      # res.insert(1, additional_inputs)
      # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    else: # no additional_inputs and no bn_additional_inputs
      # print("arrive here 3 !!!")
      # print("res: ", res)
      # print("arrive here 3-1 !!!")
      # res = [iter, ] + [one, ] + inputs_and_outputs
      res = list(res_iter_inputs) + [one, ] + [res_outputs, ]
      # res = res.insert(1, one) # body_result would be None
    # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    return res

  def new_body_fn_may16_1426pm(*carried_inputs):
    res = list(body_fn(*carried_inputs))

    # res_iter_inputs = res[:-1]
    # res_outputs = res[-1]

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
      # res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [one, ] + [res_outputs, ]
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [res_outputs, ]
      # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      # res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      # res = list(res_iter_inputs) + list(additional_inputs) + [one, ] + [res_outputs, ]
      res = list(res_iter_inputs) + list(additional_inputs) + [res_outputs, ]
      # res.insert(1, one)
      # res.insert(1, additional_inputs)
      # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    else: # no additional_inputs and no bn_additional_inputs
      # print("arrive here 3 !!!")
      # print("res: ", res)
      # print("arrive here 3-1 !!!")
      # res = [iter, ] + [one, ] + inputs_and_outputs
      # res = list(res_iter_inputs) + [one, ] + [res_outputs, ]
      res = res # list(res_iter_inputs) + [res_outputs, ]
      # res = res.insert(1, one) # body_result would be None
    # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    return res

  # def new_new_cond_fn(*carried_inputs):
  #   return cond_fn(*carried_inputs)

  # def new_new_body_fn(*carried_inputs):
  #   return body_fn(*carried_inputs)

  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32_may16_1617pm(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32_may16_1617pm(cond_fn, new_body_fn_may16_1426pm, carried_inputs, additional_inputs, bn_additional_inputs)
  return _xla_while_loop_target_second_clean_version_s32_may16_1617pm(cond_fn, body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_new_cond_fn, new_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_may16_1530pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  # iter = carried_inputs[0]
  # inputs_ouputs_val = carried_inputs[1:]

  # if bn_additional_inputs != []: 
  #   xla_device = carried_inputs[0].device
  #   # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
  #   one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device) # dtype == iter.dtype
  #   new_carried_inputs = [iter, one] + list(inputs_ouputs_val)
  #   bn_additional_inputs.insert(0, one)
  # else:
  #   new_carried_inputs = carried_inputs

  xla_device = carried_inputs[0].device
  one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device) # dtype == iter.dtype
  # new_carried_inputs = [iter, one] + list(inputs_ouputs_val)
  # bn_additional_inputs.insert(0, one)
  # else:
  #   new_carried_inputs = carried_inputs

  iter = carried_inputs[0]
  inputs_ouputs_val = carried_inputs[1:]
  new_carried_inputs = [iter, one] + list(inputs_ouputs_val)
  # new_carried_inputs = [iter, ] + list(inputs_ouputs_val)
  # if bn_additional_inputs != []:
  #   bn_additional_inputs.insert(0, one)

  # def new_cond_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
  def new_cond_fn(iter, _, *inputs_ouputs_val): # *carried_inputs):
    return cond_fn(iter, *inputs_ouputs_val)

  # print("additional_inputs: ", additional_inputs)
  # def new_body_fn(*carried_inputs):
  def new_body_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
    # res = list(body_fn(*carried_inputs))
    # return body_fn(iter, *inputs_ouputs_val)

    res = list(body_fn(iter, *inputs_ouputs_val))

    iter = res[0]
    inputs_and_outputs = res[1:]

    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
    # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
      res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
      # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      # res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      # res.insert(1, one)
      # res.insert(1, additional_inputs)
      # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    else: # no additional_inputs and no bn_additional_inputs
      # print("arrive here 3 !!!")
      # print("res: ", res)
      # print("arrive here 3-1 !!!")
      # res = [iter, ] + [one, ] + inputs_and_outputs
      res = [iter, one] + inputs_and_outputs
      # res = res.insert(1, one) # body_result would be None
    # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    return res


  def new_new_cond_fn(*carried_inputs):
    return cond_fn(*carried_inputs)

  def new_new_body_fn(*carried_inputs):
    return body_fn(*carried_inputs)


  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_new_cond_fn, new_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_pure_cond_body_fn_may16_1541pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  # xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device)

  # iter = carried_inputs[0]
  # inputs_ouputs_val = carried_inputs[1:]
  # new_carried_inputs = [iter, one] + list(inputs_ouputs_val)

  # def new_cond_fn(iter, _, *inputs_ouputs_val): # *carried_inputs):
  #   return cond_fn(iter, *inputs_ouputs_val)

  # def new_body_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
  #   # res = list(body_fn(*carried_inputs))
  #   # return body_fn(iter, *inputs_ouputs_val)

  #   res = list(body_fn(iter, *inputs_ouputs_val))

  #   iter = res[0]
  #   inputs_and_outputs = res[1:]

  #   # xla_device = carried_inputs[0].device
  #   # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
  #   # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs

  #   if additional_inputs and (bn_additional_inputs != []):
  #     print("arrive here 1 !!!")
  #     bn_additional_inputs.insert(0, one)
  #     # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
  #     res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
  #     # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
  #   elif additional_inputs and (bn_additional_inputs == []):
  #     print("arrive here 2 !!!")
  #     # res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
  #     res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
  #     # res.insert(1, one)
  #     # res.insert(1, additional_inputs)
  #     # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
  #   else: # no additional_inputs and no bn_additional_inputs
  #     # print("arrive here 3 !!!")
  #     # print("res: ", res)
  #     # print("arrive here 3-1 !!!")
  #     # res = [iter, ] + [one, ] + inputs_and_outputs
  #     res = [iter, one] + inputs_and_outputs
  #     # res = res.insert(1, one) # body_result would be None
  #   # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
  #   return res

  def new_new_cond_fn(*carried_inputs):
    return cond_fn(*carried_inputs)

  def new_new_body_fn(*carried_inputs):
    return body_fn(*carried_inputs)

  return _xla_while_loop_target_second_clean_version_s32(new_new_cond_fn, new_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_experimental(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  iter = carried_inputs[0]
  inputs_ouputs_val = carried_inputs[1:]

  if bn_additional_inputs != []: 
    xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
    one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device) # dtype == iter.dtype
    new_carried_inputs = [iter, one] + list(inputs_ouputs_val)
    bn_additional_inputs.insert(0, one)
  else:
    new_carried_inputs = carried_inputs

  # iter = carried_inputs[0]
  # inputs_ouputs_val = carried_inputs[1:]
  # new_carried_inputs = [iter, one] + list(inputs_ouputs_val)
  # new_carried_inputs = [iter, ] + list(inputs_ouputs_val)
  # if bn_additional_inputs != []:
  #   bn_additional_inputs.insert(0, one)

  # def new_cond_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
  def new_cond_fn(iter, _, *inputs_ouputs_val): # *carried_inputs):
    return cond_fn(iter, *inputs_ouputs_val)

  # print("additional_inputs: ", additional_inputs)
  # def new_body_fn(*carried_inputs):
  def new_body_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
    # res = list(body_fn(*carried_inputs))
    res = list(body_fn(iter, *inputs_ouputs_val))
    iter = res[0]
    inputs_and_outputs = res[1:]
    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
    # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs

    # if additional_inputs and (bn_additional_inputs != []):
    #   print("arrive here 1 !!!")
    #   bn_additional_inputs.insert(0, one)
    #   # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
    #   res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    #   # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    # elif additional_inputs and (bn_additional_inputs == []):
    #   print("arrive here 2 !!!")
    #   res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
    #   # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    # else: # no additional_inputs and no bn_additional_inputs
    #   # print("arrive here 3 !!!")
    #   # print("res: ", res)
    #   # print("arrive here 3-1 !!!")
    #   # res = [iter, ] + [one, ] + inputs_and_outputs
    #   res = [iter, one] + inputs_and_outputs
    res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
    return res

  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)

def _xla_while_loop_target_first_second_clean_version_s32_old(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
  one = torch.tensor(1, dtype=carried_inputs[0].dtype, device=xla_device) # dtype == iter.dtype

  iter = carried_inputs[0]
  inputs_ouputs_val = carried_inputs[1:]
  new_carried_inputs = [iter, one] + list(inputs_ouputs_val)
  # new_carried_inputs = [iter, ] + list(inputs_ouputs_val)
  # if bn_additional_inputs != []:
  #   bn_additional_inputs.insert(0, one)

  # def new_cond_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
  def new_cond_fn(iter, _, *inputs_ouputs_val): # *carried_inputs):
    return cond_fn(iter, *inputs_ouputs_val)

  # print("additional_inputs: ", additional_inputs)
  # def new_body_fn(*carried_inputs):
  def new_body_fn(iter, one, *inputs_ouputs_val): # *carried_inputs):
    # res = list(body_fn(*carried_inputs))
    res = list(body_fn(iter, *inputs_ouputs_val))
    iter = res[0]
    inputs_and_outputs = res[1:]
    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int32, device=xla_device)
    # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
      res = [iter, ] + list(additional_inputs) + bn_additional_inputs + [one, ] + inputs_and_outputs
      # res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      res = [iter, ] + list(additional_inputs) + [one, ] + inputs_and_outputs
      # res = [iter, ] + list(additional_inputs) + inputs_and_outputs
    else: # no additional_inputs and no bn_additional_inputs
      # print("arrive here 3 !!!")
      # print("res: ", res)
      # print("arrive here 3-1 !!!")
      # res = [iter, ] + [one, ] + inputs_and_outputs
      res = [iter, one] + inputs_and_outputs
    return res

  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  return _xla_while_loop_target_second_clean_version_s32(new_cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second_clean_version_s32(cond_fn, new_body_fn, new_carried_inputs, additional_inputs, bn_additional_inputs)

# used
def _xla_while_loop_target_first_second_clean_version(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  def new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
    return res

  return _xla_while_loop_target_second_clean_version(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

# commented version
def _xla_while_loop_target_first_second(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):
  def new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
    return res

  def try_new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    # res = body_fn(*carried_inputs)
    # iter = res[0]
    # inputs_and_outputs = res[1:]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    res = res + list(additional_inputs)
    # res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
    return res

  def second_try_new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    # res = body_fn(*carried_inputs)
    # iter = res[0]
    # inputs_and_outputs = res[1:]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    res = list(additional_inputs) + res
    # res = [iter,] + list(additional_inputs) + list(inputs_and_outputs,)
    return res

  def third_try_new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    res = [iter,] + inputs_items + additional_inputs + [outputs_items, ]
    return res

  def forth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, one] + inputs_items + additional_inputs + [outputs_items, ]
    return res

  def fifth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    # res = iter, 1, additiona_inputs, inputs, outputs
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, one] + additional_inputs + inputs_items + [outputs_items, ]
    return res

  # # xla_device = carried_inputs[0].device
  # xla_device = additional_inputs[0].device
  # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
  # additional_inputs = [one, ] + additional_inputs

  def sixth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    # res = iter, 1, additiona_inputs, inputs, outputs
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs) + inputs_items + [outputs_items, ]
    return res

  def seventh_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, one] + additional_inputs + inputs_items + [outputs_items, ]
    return res

  def eigth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + additional_inputs + [one, ] + inputs_items + [outputs_items, ]
    return res

  def ninth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # if not bn_additional_inputs:
    #   bn_additional_inputs = []
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_items + [outputs_items, ]
    return res

  def tenth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    # res = [iter, ] + additional_inputs + [one, ] + bn_additional_inputs + inputs_items + [outputs_items, ]
    # res = [iter, ] + additional_inputs + bn_additional_inputs[1:] + inputs_items + [outputs_items, ]
    res = [iter, ] + additional_inputs + bn_additional_inputs + inputs_items + [outputs_items, ]
    return res

  # xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
  # new_bn_additional_inputs = [one, ] + bn_additional_inputs

  def eleventh_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    inputs_items = inputs_and_outputs[:-1]
    outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # if not bn_additional_inputs:
    #   bn_additional_inputs = []
    # xla_device = carried_inputs[0].device
    # one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_items + [outputs_items, ]
    return res
  # new_additional_inputs = additional_inputs[0] + additional_inputs[1]

  def twelveth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    # inputs_items = inputs_and_outputs[:-1]
    # outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # if not bn_additional_inputs:
    #   bn_additional_inputs = []
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
    return res

  def thirteenth_try_new_body_fn(*carried_inputs):
    # add s64[] for 1
    res = list(body_fn(*carried_inputs))
    iter = res[0]
    inputs_and_outputs = res[1:]
    # inputs_items = inputs_and_outputs[:-1]
    # outputs_items = inputs_and_outputs[-1]
    # if len(inputs_and_outputs)==1:
    #   inputs_and_outputs = [inputs_and_outputs,]
    # res = res + list(additional_inputs)
    # if not bn_additional_inputs:
    #   bn_additional_inputs = []
    xla_device = carried_inputs[0].device
    one = torch.tensor(1, dtype=torch.int64, device=xla_device)
    bn_additional_inputs.insert(0, one)
    # res = [iter, ] + list(additional_inputs) + [one, ] + bn_additional_inputs + inputs_and_outputs
    res = [iter, ] + list(additional_inputs) + bn_additional_inputs + inputs_and_outputs
    return res

  # return _xla_while_loop_target(cond_fn, new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, second_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, third_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, forth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, fifth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, sixth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, seventh_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, eigth_try_new_body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target_second(cond_fn, ninth_try_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second(cond_fn, eleventh_try_new_body_fn, carried_inputs, additional_inputs, new_bn_additional_inputs)
  # return _xla_while_loop_target_second(cond_fn, twelveth_try_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  # return _xla_while_loop_target_second(cond_fn, thirteenth_try_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)
  return _xla_while_loop_target_second_clean_version(cond_fn, twelveth_try_new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)

# used new s32
def _xla_while_loop_target_second_clean_version_s32_may19_2206pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]): # bn_additional_inputs=[]: 'NoneType' object is not iterable

  #  ============================= fake_carried_inputs ==========================================  
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  # fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]
  fake_iter_input = fake_carried_inputs[:-1]
  fake_output = fake_carried_inputs[-1]

  #  ============================= additional_inputs_list_cond ======================
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))

  # modified_carried_inputs = carried_inputs + additional_inputs
  # iter = carried_inputs[0]
  # inputs = carried_inputs[1:]
  # modified_carried_inputs = [iter, ] + list(additional_inputs) + [input, ]
  # modified_carried_inputs = list(carried_inputs) + [iter, ] + list(additional_inputs) + [input, ]
  modified_carried_inputs = [iter, ] + list(additional_inputs) + [input, ]
  # print("modified_carried_inputs: ", modified_carried_inputs)
  # print("carried_inputs: ", carried_inputs)
  # print("iter: ", iter)
  # print("additional_inputs: ", additional_inputs)
  # print("input: ", input)
  modified_fake_inputs = fake_carried_inputs + fake_additiona_args

  # cond_fn get fake result first via XLA, then body_fn to unmiss the inputs in body's xlacomputation
  #  ============================= cond_fn ==========================================
  # print("carried_inputs: ", carried_inputs)
  # print("additional_inputs: ", additional_inputs)
  # modified_carried_inputs = carried_inputs + additional_inputs
  # cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_result = cond_fn(*carried_inputs, *additional_inputs) # fake one would result none input args
  # cond_result = cond_fn(*modified_carried_inputs) # fake one would result none input args
  # cond_result = cond_fn(*newest_fake_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  print("get cond_result !!!")

  #  ============================= cond ==========================================
  # print("bn_additional_inputs: ", bn_additional_inputs)
  # print("fake_additiona_args: ", fake_additiona_args)

  # fake_additiona_args += bn_additional_inputs
  # additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # additional_inputs_list_cond = fake_carried_inputs # [fake_output, ] # fake_input_output

  # additional_inputs_list_cond = modified_fake_inputs
  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  additional_inputs_list_cond = [fake_carried_inputs[0], ] + fake_additiona_args + fake_carried_inputs[1:]
  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= body_fn ==========================================
  # body_result = body_fn(*carried_inputs) # fake would miss iter
  body_result = body_fn(*carried_inputs, *additional_inputs) # fake would miss iter # right inputs
  # body_result = body_fn(carried_inputs[0], *additional_inputs, *carried_inputs[1:]) # fake would miss iter
  print("body carried_inputs: ", carried_inputs)
  print("body additional_inputs: ", additional_inputs)
  print("body inputs: ", (*carried_inputs, *additional_inputs))
  # body_result = body_fn(*newest_fake_inputs) # fake would miss iter
  # body_result = body_fn(*modified_carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  print("get body_result !!!")

  # #  ============================= cond ==========================================
  # # print("bn_additional_inputs: ", bn_additional_inputs)
  # # print("fake_additiona_args: ", fake_additiona_args)

  # # fake_additiona_args += bn_additional_inputs
  # # additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # # additional_inputs_list_cond = fake_carried_inputs # [fake_output, ] # fake_input_output
  # additional_inputs_list_cond = modified_fake_inputs
  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # cond_hlo = cond_ctx.hlo()
  # cond_computation = xb.computation_from_module_proto("condcomputation",
  #                                                     cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond computation: !!!!!!!!!")
  # print(cond_hlo_print)

  #  ============================= body xlacomputation ==========================================
  # additional_inputs_list_body = fake_carried_inputs # [fake_output, ] # fake_input_output
  additional_inputs_list_body = modified_fake_inputs
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), modified_carried_inputs)
  # body_ctx.buildforiloop(list(body_result), [])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # #  ============================= cond ==========================================
  # fake_additiona_args += bn_additional_inputs
  # additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_body)
  # cond_hlo = cond_ctx.hlo()
  # cond_computation = xb.computation_from_module_proto("condcomputation",
  #                                                     cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond computation: !!!!!!!!!")
  # print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)
  print("total_inputs: ", total_inputs)

  print("get total_inputs !!!")

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result


def _xla_while_loop_target_second_clean_version_s32_may17_1528pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]): # bn_additional_inputs=[]: 'NoneType' object is not iterable

  #  ============================= fake_carried_inputs ==========================================  
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  # fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]
  fake_iter_input = fake_carried_inputs[:-1]
  fake_output = fake_carried_inputs[-1]

  #  ============================= additional_inputs_list_cond ======================
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))

  # modified_carried_inputs = carried_inputs + additional_inputs
  # iter = carried_inputs[0]
  # inputs = carried_inputs[1:]
  # modified_carried_inputs = [iter, ] + list(additional_inputs) + [input, ]
  # modified_carried_inputs = list(carried_inputs) + [iter, ] + list(additional_inputs) + [input, ]
  modified_carried_inputs = [iter, ] + list(additional_inputs) + [input, ]
  # print("modified_carried_inputs: ", modified_carried_inputs)
  # print("carried_inputs: ", carried_inputs)
  # print("iter: ", iter)
  # print("additional_inputs: ", additional_inputs)
  # print("input: ", input)
  modified_fake_inputs = fake_carried_inputs + fake_additiona_args

  # cond_fn get fake result first via XLA, then body_fn to unmiss the inputs in body's xlacomputation
  #  ============================= cond_fn ==========================================
  # print("carried_inputs: ", carried_inputs)
  # print("additional_inputs: ", additional_inputs)
  # modified_carried_inputs = carried_inputs + additional_inputs
  # cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_result = cond_fn(*carried_inputs, *additional_inputs) # fake one would result none input args
  # cond_result = cond_fn(*modified_carried_inputs) # fake one would result none input args
  # cond_result = cond_fn(*newest_fake_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  print("get cond_result !!!")

  #  ============================= body_fn ==========================================
  # body_result = body_fn(*carried_inputs) # fake would miss iter
  body_result = body_fn(*carried_inputs, *additional_inputs) # fake would miss iter
  # body_result = body_fn(*newest_fake_inputs) # fake would miss iter
  # body_result = body_fn(*modified_carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  print("get body_result !!!")

  #  ============================= cond ==========================================
  # print("bn_additional_inputs: ", bn_additional_inputs)
  # print("fake_additiona_args: ", fake_additiona_args)

  # fake_additiona_args += bn_additional_inputs
  # additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # additional_inputs_list_cond = fake_carried_inputs # [fake_output, ] # fake_input_output
  additional_inputs_list_cond = modified_fake_inputs
  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= body xlacomputation ==========================================
  # additional_inputs_list_body = fake_carried_inputs # [fake_output, ] # fake_input_output
  additional_inputs_list_body = modified_fake_inputs
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), modified_carried_inputs)
  # body_ctx.buildforiloop(list(body_result), [])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # #  ============================= cond ==========================================
  # fake_additiona_args += bn_additional_inputs
  # additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_body)
  # cond_hlo = cond_ctx.hlo()
  # cond_computation = xb.computation_from_module_proto("condcomputation",
  #                                                     cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond computation: !!!!!!!!!")
  # print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)
  print("total_inputs: ", total_inputs)

  print("get total_inputs !!!")

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

def _xla_while_loop_target_second_clean_version_s32_may17_1018am(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=None):

  #  ============================= fake_carried_inputs ==========================================  
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  # fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]
  fake_iter_input = fake_carried_inputs[:-1]
  fake_output = fake_carried_inputs[-1]

  # for i in range(len(fake_carried_inputs)): print("fake_carried_inputs: ", i, "size: ", fake_carried_inputs[i].size())

  # cond_fn get fake result first via XLA, then body_fn to unmiss the inputs in body's xlacomputation
  #  ============================= cond_fn ==========================================
  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  #  ============================= body_fn ==========================================
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  #  ============================= body xlacomputation ==========================================
  additional_inputs_list_body = fake_carried_inputs # [fake_output, ] # fake_input_output
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  #  ============================= additional_inputs_list_cond ======================
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))
  # print("print fake_additiona_args !!!")
  # for i in range(len(fake_additiona_args)): print("fake_additiona_args: ", i, " size: ", fake_additiona_args[i].size())
  # print("after print fake_additiona_args !!!")

  #  ============================= cond ==========================================
  fake_additiona_args += bn_additional_inputs
  additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_ctx.buildforiloop([cond_result], additional_inputs_list_body)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

def _xla_while_loop_target_second_clean_version_s32_may16_2138pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=None):

  # print("type carried_inputs: ", type(carried_inputs))
  # print("type carried_inputs[0]: ", type(carried_inputs[0]))
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  # fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]
  fake_iter_input = fake_carried_inputs[:-1]
  fake_output = fake_carried_inputs[-1]

  # cond_fn get fake result first via XLA, then body_fn to unmiss the inputs in body's xlacomputation
  #  ============================= cond_fn ==========================================
  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  #  ============================= body_fn ==========================================
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  #  ============================= additional_inputs_list_cond ======================
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))
  print("print fake_additiona_args !!!")
  for i in range(len(fake_additiona_args)): print("fake_additiona_args: ", i, " size: ", fake_additiona_args[i].size())
  print("after print fake_additiona_args !!!")

  #  ============================= body xlacomputation ==========================================
  # body_result = body_fn(*carried_inputs) # fake would miss iter
  # body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  # body_ctx.set_name_string("bodyctx")
  # additional_inputs_list_body = [fake_output, ] # fake_input_output

  # check index with lenth of carried_inputs, if less, that means we missed some unused args, and these unused args could be output_placeholder, or bn_weight_bias

  additional_inputs_list_body = [fake_output, ] # fake_input_output
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  #  ============================= cond ==========================================
  # #  === additional_inputs_list_cond ===
  # fake_additiona_args = []
  # for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
  #   device = additional_input.device
  #   fake_additiona_args.append(
  #       torch.randint(
  #           10, additional_input.size(),
  #           dtype=additional_input.dtype).to(device))
  # #  === add one ===
  # one = torch.tensor(1, dtype=torch.int32, device=device)
  # bn_additional_inputs.insert(0, one)
  #  === add bn_additional_inputs ===
  fake_additiona_args += bn_additional_inputs
  additional_inputs_list_cond = fake_additiona_args + fake_input_output

  # cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  # cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  # cond_ctx.set_name_string("condctx")

  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

def _xla_while_loop_target_second_clean_version_s32_may16_1617pm(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=None):

  # print("type carried_inputs: ", type(carried_inputs))
  # print("type carried_inputs[0]: ", type(carried_inputs[0]))
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  # fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]
  fake_iter_input = fake_carried_inputs[:-1]
  fake_output = fake_carried_inputs[-1]

  #  ============================= body ==========================================
  # additional_inputs_list_body = fake_input_output
  # for i in range(len(carried_inputs)): print("carried_inputs: ", i, " size: ", carried_inputs[i].size())
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  additional_inputs_list_body = [fake_output, ] # fake_input_output

  # print("arrive here 4 !!!")
  # print("body_result: ", body_result)
  # print("arrive here 4-1 !!!")
  # print("additional_inputs_list_body: ", additional_inputs_list_body)
  # print("arrive here 4-2 !!!")

  # print("body_result: ", body_result)
  # print("additional_inputs_list_body: ", additional_inputs_list_body)
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), ())
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # iter, input, weght, bias, s64[], bn_weight, bias, output
  # iter, input, weight, bias, s64[]. ouput

  #  ============================= cond ==========================================
  #  === additional_inputs_list_cond ===
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))
  # #  === add one ===
  # one = torch.tensor(1, dtype=torch.int32, device=device)
  # bn_additional_inputs.insert(0, one)
  #  === add bn_additional_inputs ===
  fake_additiona_args += bn_additional_inputs
  additional_inputs_list_cond = fake_additiona_args + fake_input_output

  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

def _xla_while_loop_target_second_clean_version_s32(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=None):

  # print("type carried_inputs: ", type(carried_inputs))
  # print("type carried_inputs[0]: ", type(carried_inputs[0]))
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]

  #  ============================= body ==========================================
  # additional_inputs_list_body = fake_input_output
  # for i in range(len(carried_inputs)): print("carried_inputs: ", i, " size: ", carried_inputs[i].size())
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  additional_inputs_list_body = fake_input_output

  # print("arrive here 4 !!!")
  # print("body_result: ", body_result)
  # print("arrive here 4-1 !!!")
  # print("additional_inputs_list_body: ", additional_inputs_list_body)
  # print("arrive here 4-2 !!!")

  # print("body_result: ", body_result)
  # print("additional_inputs_list_body: ", additional_inputs_list_body)
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), ())
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # iter, input, weght, bias, s64[], bn_weight, bias, output
  # iter, input, weight, bias, s64[]. ouput

  #  ============================= cond ==========================================
  #  === additional_inputs_list_cond ===
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))
  # #  === add one ===
  # one = torch.tensor(1, dtype=torch.int32, device=device)
  # bn_additional_inputs.insert(0, one)
  #  === add bn_additional_inputs ===
  fake_additiona_args += bn_additional_inputs
  additional_inputs_list_cond = fake_additiona_args + fake_input_output

  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

# used 
def _xla_while_loop_target_second_clean_version(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=None):

  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]

  #  ============================= body ==========================================
  additional_inputs_list_body = fake_input_output
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # iter, input, weght, bias, s64[], bn_weight, bias, output
  # iter, input, weight, bias, s64[]. ouput

  #  ============================= cond ==========================================
  #  === additional_inputs_list_cond ===
  fake_additiona_args = []
  for additional_input in additional_inputs: # additional_inputs would has value after body_fn been traced
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))
  #  === add one ===
  one = torch.tensor(1, dtype=torch.int64, device=device)
  bn_additional_inputs.insert(0, one)
  #  === add bn_additional_inputs ===
  fake_additiona_args += bn_additional_inputs
  additional_inputs_list_cond = fake_additiona_args + fake_input_output

  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  #  ============================= xla::while ==========================================
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(input_and_outputs_value)

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

# commented version
def _xla_while_loop_target_second(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=None):
  # print("arrive _xla_while_loop")
  # print("carried_inputs: ", carried_inputs)
  # for i in range(len(carried_inputs)): print("carried_inputs ", i, " size: ", carried_inputs[i].size())
  # print("type carried_inputs: ", type(carried_inputs))
  # for i in range(len(additional_inputs)): print("additional_inputs ", i, " size: ", additional_inputs[i].size())
  # print("additional_inputs: ", additional_inputs)
  ### use output as input now case, so we could get output in the return value from original inpjut position

  # for i in range(len(bn_additional_inputs)): print("bn_additional_inputs ", i, " size: ", bn_additional_inputs[i].size())

  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]

  # for i in range(len(fake_carried_inputs)): print("fake_carried_inputs ", i, " size: ", fake_carried_inputs[i].size())

  # fake_additiona_args = [] # fake_carried_inputs
  # for additional_input in additional_inputs:
  #   print("arrive here for confirmation !!!")
  #   device = additional_input.device
  #   fake_additiona_args.append(
  #       torch.randint(
  #           10, additional_input.size(),
  #           dtype=additional_input.dtype).to(device))

  # for i in range(len(fake_carried_inputs)): print("second fake_carried_inputs ", i, " size: ", fake_carried_inputs[i].size())
  # print("fake_carried_inputs: ", fake_carried_inputs)

  # cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  # cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  # cond_ctx.set_name_string("condctx")

  #  ============================= body ==========================================
  # body first before cond, after body, the additional_inputs would have expected value
  # generate body_fn xlacomputation
  # body_result = body_fn(*fake_carried_inputs)
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  # for i in range(len(body_result)): print("body_result ", i, "size: ", body_result[i].size())

  # # add output arg in body's input for result save to meet requirement
  # # TODO(@manfei): treat hard-code body xlacomputation change: currently add non-changed output_value argument if additional_inputs(weight/bias) exists
  # if additional_inputs:
  #   # print("arrive here !!!")
  #   additional_inputs_list_body = [fake_carried_inputs[5]]
  # else:
  #   # print("arrive here too !!!")
  #   additional_inputs_list_body = []

  # # for no-weight-bias-return body, we need to add params in build
  # additional_inputs_list_body = additional_inputs
  # additional_inputs_list_body = [fake_input_output[-1],]
  additional_inputs_list_body = fake_input_output
  # print("fake_iter: ", fake_iter)
  # print("fake_input_output: ", fake_input_output)
  # print("additional_inputs_list_body: ", additional_inputs_list_body)

  # TODO(@manfei): treat hard-code parameters: additional_inputs_list_body
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), [])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  # body_hlo_print = xb.get_computation_hlo(body_computation)
  # print("body computation: !!!!!!!!!")
  # print(body_hlo_print)

  # for i in range(len(additional_inputs)): print("4 additional_inputs ", i, " size: ", additional_inputs[i].size())

  #  ============================= cond ==========================================
  fake_additiona_args = [] # fake_carried_inputs
  for additional_input in additional_inputs:
    # print("arrive here for confirmation !!!")
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))

  # === add one ===
  # xla_device = carried_inputs[0].device
  one = torch.tensor(1, dtype=torch.int64, device=device) # xla_device)
  # fake_additiona_args.append(one)
  bn_additional_inputs.insert(0, one)

  # === add bn_additional_inputs ===
  fake_additiona_args += bn_additional_inputs

  # for i in range(len(bn_additional_inputs)): print("bn_additional_inputs ", i, " size: ", bn_additional_inputs[i].size())

  # for i in range(len(additional_inputs)): print("additional_inputs ", i, " size: ", additional_inputs[i].size())

  # TODO(@manfei): specify which element is for which argument like a,b,c
  # cond_result = cond_fn(*fake_carried_inputs)
  # cond_result = cond_fn(*fake_carried_inputs_all_args)
  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  # # skip exist iter, add other additional inputs
  # additional_inputs_list_cond = list(
  #     fake_carried_inputs_all_args[1:] # fake_carried_inputs[1:]
  # )
  additional_inputs_list_cond = fake_additiona_args + fake_input_output # due to post order

  # # seems due to post-order, input was in the final position of xlacomputation, so move inputs' order like that too
  # if additional_inputs:
  #   print("arrive here for cond !!!")
  #   tmp_output = additional_inputs_list_cond[0]  # not used, change order doesn't affect logic
  #   del additional_inputs_list_cond[0]  # not used, change order doesn't affect logic
  #   additional_inputs_list_cond.append(tmp_output)  # not used, change order doesn't affect logic

  # print("additional_inputs_list_cond: ", additional_inputs_list_cond)
  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # cond_ctx.buildforiloop([cond_result], ())
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond computation: !!!!!!!!!")
  # print(cond_hlo_print)

  # for i in range(len(additional_inputs)): print("3 additional_inputs ", i, " size: ", additional_inputs[i].size())


  # # generate body_fn xlacomputation
  # # body_result = body_fn(*fake_carried_inputs)
  # body_result = body_fn(*carried_inputs) # fake would miss iter
  # body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  # body_ctx.set_name_string("bodyctx")

  # # # add output arg in body's input for result save to meet requirement
  # # # TODO(@manfei): treat hard-code body xlacomputation change: currently add non-changed output_value argument if additional_inputs(weight/bias) exists
  # # if additional_inputs:
  # #   # print("arrive here !!!")
  # #   additional_inputs_list_body = [fake_carried_inputs[5]]
  # # else:
  # #   # print("arrive here too !!!")
  # #   additional_inputs_list_body = []

  # # # for no-weight-bias-return body, we need to add params in build
  # # additional_inputs_list_body = additional_inputs
  # additional_inputs_list_body = [fake_input_output[-1],]

  # # TODO(@manfei): treat hard-code parameters: additional_inputs_list_body
  # body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # # body_ctx.buildforiloop(list(body_result), ())
  # body_hlo = body_ctx.hlo()
  # body_computation = xb.computation_from_module_proto("bodycomputation",
  #                                                     body_hlo)
  # # body_hlo_print = xb.get_computation_hlo(body_computation)
  # # print("body computation: !!!!!!!!!")
  # # print(body_hlo_print)

  # # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
  # total_inputs = carried_inputs + tuple(additional_inputs)
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]

  # # === add one ===
  # # xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=torch.int64, device=device) # xla_device)
  # fake_additiona_args.append(one)

  # total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple([one, ]) + tuple(bn_additional_inputs) + tuple(carried_inputs[1:])
  # total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple([one,]) + tuple(bn_additional_inputs) + tuple(carried_inputs[1:])
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(bn_additional_inputs) + tuple(carried_inputs[1:])

  # for i in range(len(carried_inputs)): print("2 carried_inputs ", i, " size: ", carried_inputs[i].size())
  # for i in range(len(additional_inputs)): print("2 additional_inputs ", i, " size: ", additional_inputs[i].size())

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)
  # print("params: ", params)

  # # change order of output of real inputs to match the real body's xlacomputation due to post order
  # # TODO(@manfei): treat hard-code input arguments, currently switch bias and output_value if additional_inputs(weight/bias) exists
  # if additional_inputs:
  #   tmp_output = params[2] # 1]
  #   del params[2] # 1]
  #   params.append(tmp_output)
  #   # tmp_bias = params[-3]
  #   # del params[-3]
  #   # params.append(tmp_bias)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result


def _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  # print("arrive _xla_while_loop")
  # print("carried_inputs: ", carried_inputs)
  # for i in range(len(carried_inputs)): print("carried_inputs ", i, " size: ", carried_inputs[i].size())
  # print("type carried_inputs: ", type(carried_inputs))
  # for i in range(len(additional_inputs)): print("additional_inputs ", i, " size: ", additional_inputs[i].size())
  # print("additional_inputs: ", additional_inputs)
  ### use output as input now case, so we could get output in the return value from original inpjut position

  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  fake_iter = fake_carried_inputs[0]
  fake_input_output = fake_carried_inputs[1:]

  # for i in range(len(fake_carried_inputs)): print("fake_carried_inputs ", i, " size: ", fake_carried_inputs[i].size())

  # fake_additiona_args = [] # fake_carried_inputs
  # for additional_input in additional_inputs:
  #   print("arrive here for confirmation !!!")
  #   device = additional_input.device
  #   fake_additiona_args.append(
  #       torch.randint(
  #           10, additional_input.size(),
  #           dtype=additional_input.dtype).to(device))

  # for i in range(len(fake_carried_inputs)): print("second fake_carried_inputs ", i, " size: ", fake_carried_inputs[i].size())
  # print("fake_carried_inputs: ", fake_carried_inputs)

  # cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  # cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  # cond_ctx.set_name_string("condctx")

  #  ============================= body ==========================================
  # body first before cond, after body, the additional_inputs would have expected value
  # generate body_fn xlacomputation
  # body_result = body_fn(*fake_carried_inputs)
  body_result = body_fn(*carried_inputs) # fake would miss iter
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  for i in range(len(body_result)): print("body_result ", i, "size: ", body_result[i].size())

  # # add output arg in body's input for result save to meet requirement
  # # TODO(@manfei): treat hard-code body xlacomputation change: currently add non-changed output_value argument if additional_inputs(weight/bias) exists
  # if additional_inputs:
  #   # print("arrive here !!!")
  #   additional_inputs_list_body = [fake_carried_inputs[5]]
  # else:
  #   # print("arrive here too !!!")
  #   additional_inputs_list_body = []

  # # for no-weight-bias-return body, we need to add params in build
  # additional_inputs_list_body = additional_inputs
  # additional_inputs_list_body = [fake_input_output[-1],]
  additional_inputs_list_body = fake_input_output
  # print("fake_iter: ", fake_iter)
  # print("fake_input_output: ", fake_input_output)
  # print("additional_inputs_list_body: ", additional_inputs_list_body)

  # TODO(@manfei): treat hard-code parameters: additional_inputs_list_body
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), [])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # for i in range(len(additional_inputs)): print("4 additional_inputs ", i, " size: ", additional_inputs[i].size())

  #  ============================= cond ==========================================
  fake_additiona_args = [] # fake_carried_inputs
  for additional_input in additional_inputs:
    # print("arrive here for confirmation !!!")
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))

  # # === add one ===
  # # xla_device = carried_inputs[0].device
  # one = torch.tensor(1, dtype=torch.int64, device=device) # xla_device)
  # fake_additiona_args.append(one)

  # === add 

  # TODO(@manfei): specify which element is for which argument like a,b,c
  # cond_result = cond_fn(*fake_carried_inputs)
  # cond_result = cond_fn(*fake_carried_inputs_all_args)
  cond_result = cond_fn(*carried_inputs) # fake one would result none input args
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  # # skip exist iter, add other additional inputs
  # additional_inputs_list_cond = list(
  #     fake_carried_inputs_all_args[1:] # fake_carried_inputs[1:]
  # )
  additional_inputs_list_cond = fake_additiona_args + fake_input_output # due to post order

  # # seems due to post-order, input was in the final position of xlacomputation, so move inputs' order like that too
  # if additional_inputs:
  #   print("arrive here for cond !!!")
  #   tmp_output = additional_inputs_list_cond[0]  # not used, change order doesn't affect logic
  #   del additional_inputs_list_cond[0]  # not used, change order doesn't affect logic
  #   additional_inputs_list_cond.append(tmp_output)  # not used, change order doesn't affect logic

  # print("additional_inputs_list_cond: ", additional_inputs_list_cond)
  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # cond_ctx.buildforiloop([cond_result], ())
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  # cond_hlo_print = xb.get_computation_hlo(cond_computation)
  # print("cond computation: !!!!!!!!!")
  # print(cond_hlo_print)

  # for i in range(len(additional_inputs)): print("3 additional_inputs ", i, " size: ", additional_inputs[i].size())


  # # generate body_fn xlacomputation
  # # body_result = body_fn(*fake_carried_inputs)
  # body_result = body_fn(*carried_inputs) # fake would miss iter
  # body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  # body_ctx.set_name_string("bodyctx")

  # # # add output arg in body's input for result save to meet requirement
  # # # TODO(@manfei): treat hard-code body xlacomputation change: currently add non-changed output_value argument if additional_inputs(weight/bias) exists
  # # if additional_inputs:
  # #   # print("arrive here !!!")
  # #   additional_inputs_list_body = [fake_carried_inputs[5]]
  # # else:
  # #   # print("arrive here too !!!")
  # #   additional_inputs_list_body = []

  # # # for no-weight-bias-return body, we need to add params in build
  # # additional_inputs_list_body = additional_inputs
  # additional_inputs_list_body = [fake_input_output[-1],]

  # # TODO(@manfei): treat hard-code parameters: additional_inputs_list_body
  # body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # # body_ctx.buildforiloop(list(body_result), ())
  # body_hlo = body_ctx.hlo()
  # body_computation = xb.computation_from_module_proto("bodycomputation",
  #                                                     body_hlo)
  # # body_hlo_print = xb.get_computation_hlo(body_computation)
  # # print("body computation: !!!!!!!!!")
  # # print(body_hlo_print)

  # # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
  # total_inputs = carried_inputs + tuple(additional_inputs)
  iter_value = carried_inputs[0]
  input_and_outputs_value = carried_inputs[1:]
  total_inputs = tuple([iter_value,]) + tuple(additional_inputs) + tuple(carried_inputs[1:])

  # for i in range(len(carried_inputs)): print("2 carried_inputs ", i, " size: ", carried_inputs[i].size())
  # for i in range(len(additional_inputs)): print("2 additional_inputs ", i, " size: ", additional_inputs[i].size())

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)
  # print("params: ", params)

  # # change order of output of real inputs to match the real body's xlacomputation due to post order
  # # TODO(@manfei): treat hard-code input arguments, currently switch bias and output_value if additional_inputs(weight/bias) exists
  # if additional_inputs:
  #   tmp_output = params[2] # 1]
  #   del params[2] # 1]
  #   params.append(tmp_output)
  #   # tmp_bias = params[-3]
  #   # del params[-3]
  #   # params.append(tmp_bias)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

def _xla_while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  # print("arrive _xla_while_loop")

  # print("carried_inputs: ", carried_inputs)
  print("--- --- --- carried_inputs --- --- ---")
  for i in range(len(carried_inputs)):
    print("carried_inputs ", i, " size: ", carried_inputs[i].size())
  print("--- --- --- additional_inputs --- --- ---")
  # print("additional_inputs: ", additional_inputs)
  for i in range(len(additional_inputs)):
    print("additional_inputs ", i, " size: ", additional_inputs[i].size())

  # print("carried_inputs: ", carried_inputs)
  # print("additional_inputs: ", additional_inputs)
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  for additional_input in additional_inputs:
    device = additional_input.device
    fake_carried_inputs.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))

  # TODO(@manfei): specify which element is for which argument like a,b,c
  cond_result = cond_fn(*fake_carried_inputs)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  # TODO(@manfei): treat hard-code cond xlacomputation change: currently switch output_value and weight position if additional_inputs(weight/bias) exists
  additional_inputs_list_cond = list(
      fake_carried_inputs[2:]
  )  # all missed arguments except upper/lower due to PyTorch/XLA trace from output tensor
  # reorder the additional_inputs due to the given additional_inputs are not generated with expected order, let's check how `additional_inputs` was generated for mnist
  if additional_inputs:
    # print("arrive here for cond !!!")
    tmp_output = additional_inputs_list_cond[3]  # not used, change order doesn't affect logic
    del additional_inputs_list_cond[3]  # not used, change order doesn't affect logic
    additional_inputs_list_cond.append(tmp_output)  # not used, change order doesn't affect logic

  cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
  # cond_ctx.buildforiloop([cond_result], ())
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  # generate body_fn xlacomputation
  body_result = body_fn(*fake_carried_inputs)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  # TODO(@manfei): treat hard-code body xlacomputation change: currently add non-changed output_value argument if additional_inputs(weight/bias) exists
  if additional_inputs:
    # print("arrive here !!!")
    additional_inputs_list_body = [fake_carried_inputs[5]]
  else:
    # print("arrive here too !!!")
    additional_inputs_list_body = []

  # TODO(@manfei): treat hard-code parameters: additional_inputs_list_body
  body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
  # body_ctx.buildforiloop(list(body_result), ())
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
  total_inputs = carried_inputs + additional_inputs
  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # TODO(@manfei): treat hard-code input arguments, currently switch bias and output_value if additional_inputs(weight/bias) exists
  if additional_inputs:
    tmp_output = params[5]
    del params[5]
    params.append(tmp_output)
    # tmp_bias = params[-3]
    # del params[-3]
    # params.append(tmp_bias)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (total_inputs), computation)

  return result

# def _xla_while_loop_get_xla_computation(cond_fn, body_fn, carried_inputs, additional_inputs=None):
#   # fake carried_inputs to split formal code
#   fake_carried_inputs = []
#   for carried_input in carried_inputs:
#     device = carried_input.device
#     fake_carried_inputs.append(
#         torch.randint(10, carried_input.size(),
#                       dtype=carried_input.dtype).to(device))
#   for additional_input in additional_inputs:
#     device = additional_input.device
#     fake_carried_inputs.append(
#         torch.randint(
#             10, additional_input.size(),
#             dtype=additional_input.dtype).to(device))

#   # cond_fn xlacomputation
#   additional_inputs_list_cond = list(
#       fake_carried_inputs[2:]
#   )  # all missed arguments except upper/lower due to PyTorch/XLA trace from output tensor
#   if additional_inputs:
#     tmp_bias = additional_inputs_list_cond[
#         -3]  # not used, change order doesn't affect logic
#     del additional_inputs_list_cond[
#         -3]  # not used, change order doesn't affect logic
#     additional_inputs_list_cond.append(
#         tmp_bias)  # not used, change order doesn't affect logic

#   cond_result = cond_fn(*fake_carried_inputs)
#   # cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
#   # cond_ctx.set_name_string("condctx")
#   # cond_ctx.buildforiloop([cond_result], additional_inputs_list_cond)
#   # cond_hlo = cond_ctx.hlo()
#   # cond_computation = xb.computation_from_module_proto("condcomputation",
#   #                                                     cond_hlo)
#   cond_computation = torch_xla._XLAC._get_xla_computation([cond_result], [], True)
#   cond_hlo_print = xb.get_computation_hlo(cond_computation)
#   print("cond computation: !!!!!!!!!")
#   print(cond_hlo_print)

#   # generate body_fn xlacomputation
#   if additional_inputs:
#     additional_inputs_list_body = [fake_carried_inputs[-3]]
#   else:
#     additional_inputs_list_body = []

#   body_result = body_fn(*fake_carried_inputs)
#   # body_ctx = torch_xla._XLAC.lowering.LoweringContext()
#   # body_ctx.set_name_string("bodyctx")
#   # body_ctx.buildforiloop(list(body_result), additional_inputs_list_body)
#   # body_hlo = body_ctx.hlo()
#   # body_computation = xb.computation_from_module_proto("bodycomputation",
#   #                                                     body_hlo)
#   body_computation = torch_xla._XLAC._get_xla_computation(list(body_result), [], True)
#   body_hlo_print = xb.get_computation_hlo(body_computation)
#   print("body computation: !!!!!!!!!")
#   print(body_hlo_print)

#   # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
#   total_inputs = carried_inputs + additional_inputs
#   kwargs = {}
#   if type(total_inputs) is tuple:
#     shapes = xb.tensor_shape(total_inputs)
#   else:
#     shapes = xb.tensor_shape((total_inputs))
#   builder = xb.create_builder('test_while')
#   params = []
#   for shape in shapes:
#     p = xb.mkparam(builder, len(params), shape)
#     params.append(p)

#   # TODO(@manfei): treat hard-code input arguments, currently switch bias and output_value if additional_inputs(weight/bias) exists
#   if additional_inputs:
#     tmp_bias = params[-3]
#     del params[-3]
#     params.append(tmp_bias)

#   # generate while xlacomputation
#   input_tuple = xb.Op.tuple(tuple(params))
# 	# @@ -94,6 +388,6 @@ def _xla_while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs):

#   # gain final result with generated while xlacomputation
#   result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
#                                                  (total_inputs), computation)

#   return result


