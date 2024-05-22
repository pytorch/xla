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


def fori_loop(upper, lower, body_fun, *input_value):

  device = xm.xla_device()
  if (upper < lower):
    print("ERROR: upper should be a larger number than lower")
  iteri = upper - lower

  def cond_fn(iteri, *input_value): # make sure name of variable match
    return iteri > 0

  def new_body_fn(iteri, *input_value):
    return iteri - 1, body_fun(*input_value)

  inputs = (iteri, ) + input_value
  # print("inputs: ", inputs)
  # res =  while_loop(cond_fn, body_fn, (iteri, *input_value))
  res =  while_loop(cond_fn, new_body_fn, inputs)

  return res

  # one_value = torch.tensor([1], dtype=torch.int32, device=device)
  # if (hasattr(body_fun, 'weight') or hasattr(body_fun, 'bias')):
  #   output_value = torch.zeros([20], dtype=torch.float32, device=device)

  #   def cond_fn(upper, lower, one_value, x, input_value, output_value):
  #     return lower[0] < upper[0]

  #   def body_fn(upper, lower, one_value, x, input_value, output_value):
  #     new_lower = torch.add(one_value, lower)
  #     output_value = body_fun(input_value)
  #     weight = body_fun.weight  # not be used actually, initialized as placeholder xlacomputation requirement
  #     bias = body_fun.bias  # not be used actually, initialized as placeholder xlacomputation requirement
  #     return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
  #         one_value, x), input_value.clone(), bias.clone(), weight.clone(
  #         ), output_value.clone()

  #   res = torch_while_loop(
  #       cond_fn, body_fn,
  #       (upper, lower, one_value, init_val, input_value, output_value))
  # else:
  #   output_value = torch.tensor([1], dtype=torch.int32, device=device)

  #   def cond_fn(upper, lower, one_value, x, input_value):
  #     return lower[0] < upper[0]

  #   def body_fn(upper, lower, one_value, x, input_value):
  #     new_lower = torch.add(one_value, lower)
  #     output_val = body_fun(one_value, input_value)
  #     return upper.clone(), new_lower.clone(), one_value.clone(), torch.add(
  #         one_value, x), output_val.clone()

  #   res = torch_while_loop(cond_fn, body_fn,
  #                          (upper, lower, one_value, init_val, input_value))

  # return res


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  if additional_inputs is None:
    additional_inputs = tuple()
  return _xla_while_loop_wrapper(cond_fn, body_fn, carried_inputs, additional_inputs)


def _xla_while_loop_wrapper(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]):

  # print("additional_inputs: ", additional_inputs)
  # for i in range(len(additional_inputs)): print("additional_inputs [", i, "][0] : ", additional_inputs[i][0])
  def new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))

    if additional_inputs and (bn_additional_inputs != []):
      print("arrive here 1 !!!")
      bn_additional_inputs.insert(0, one)
      res = list(res_iter_inputs) + list(additional_inputs) + bn_additional_inputs + [res_outputs, ]
    elif additional_inputs and (bn_additional_inputs == []):
      print("arrive here 2 !!!")
      res = [res[0], ] + list(additional_inputs) + res[1:]
    else:
      print("arrive here 3 !!!")
      res = res
    return res

  return _xla_while_loop(cond_fn, new_body_fn, carried_inputs, additional_inputs, bn_additional_inputs)


def _xla_while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=None, bn_additional_inputs=[]): # bn_additional_inputs=[]: 'NoneType' object is not iterable

  #  ====== fake_carried_inputs ======
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))

  fake_input_output = fake_carried_inputs[1:]
  fake_iter_input = fake_carried_inputs[:-1]
  fake_output = fake_carried_inputs[-1]

  #  ====== additional_inputs_list_cond ======
  fake_additiona_args = []
  for additional_input in additional_inputs:
    device = additional_input.device
    fake_additiona_args.append(
        torch.randint(
            10, additional_input.size(),
            dtype=additional_input.dtype).to(device))

  #  ====== additional_inputs_list_cond ======
  dummy_inputs_list = [fake_carried_inputs[0], ] + fake_additiona_args + fake_carried_inputs[1:]

  body_result = body_fn(carried_inputs[0], *fake_carried_inputs[1:], *additional_inputs)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  #  ====== body xlacomputation ======
  body_ctx.buildforiloop(list(body_result), dummy_inputs_list)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)

  #  ====== cond_fn ======
  cond_result = cond_fn(*carried_inputs, *additional_inputs)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")

  #  ====== cond xlacomputation ======
  cond_ctx.buildforiloop([cond_result], dummy_inputs_list)
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)

  #  ====== xla::while ======
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

  # return result

  # print("result size: ", result.size())
  # print("result size: ", result)
  # for i in range(len(result)): print(" result ", i, " size: ", result[i].size())
  # for i in range(len(result)): print(" result ", i, " : ", result[i])

  # unwrapper result without additional_inputs and bn_additional_inputs
  # res = [res[0], ] + list(additional_inputs) + res[1:]
  additional_inputs_len = len(additional_inputs) + 1
  # print("additional_inputs_len: ", additional_inputs_len)
  final_res = [result[0], ] + result[additional_inputs_len:]

  return final_res
