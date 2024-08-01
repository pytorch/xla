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
from torch._higher_order_ops.utils import _has_potential_branch_input_mutation


def fori_loop(lower, upper, body_fun, *input_value):

  device = xm.xla_device()
  if (upper < lower):
    print("ERROR: upper should be a larger number than lower")
  iteri = upper - lower

  def cond_fn(iteri, *input_value):
    return iteri > 0

  def new_body_fn(iteri, *input_value):
    return iteri - 1, body_fun(*input_value)

  inputs = (iteri,) + input_value
  res = _xla_while_loop_wrapper(
      cond_fn, new_body_fn, inputs, (), fake_tensor=True)

  return res


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  if additional_inputs is None:
    additional_inputs = tuple()
  return _xla_while_loop_wrapper(cond_fn, body_fn, carried_inputs,
                                 additional_inputs)


def _xla_while_loop_wrapper(cond_fn,
                            body_fn,
                            carried_inputs,
                            additional_inputs=None,
                            fake_tensor=False):

  def new_body_fn(*carried_inputs):
    res = list(body_fn(*carried_inputs))
    if additional_inputs:
      res = [
          res[0],
      ] + list(additional_inputs) + res[1:]
    else:
      res = res
    return res

  return _xla_while_loop(cond_fn, new_body_fn, carried_inputs,
                         additional_inputs, fake_tensor)


def _xla_while_loop(cond_fn,
                    body_fn,
                    carried_inputs,
                    additional_inputs=None,
                    fake_tensor=False):

  #  ====== fake_carried_inputs ======
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    fake_carried_inputs.append(
        torch.empty(carried_input.size(), dtype=carried_input.dtype).to(device))

  #  ====== additional_inputs_list_cond ======
  fake_additiona_args = []
  for additional_input in additional_inputs:
    device = additional_input.device
    fake_additiona_args.append(
        torch.empty(additional_input.size(),
                    dtype=additional_input.dtype).to(device))

  #  ====== inputs_list ======
  #  specify body_fn_inputs/cond_fn_inputs, and add caught additional_inputs into fn_inputs
  if additional_inputs or fake_tensor:
    # replace inputs(carried_inputs[1:]) with fake tensors to fix missed arguments problem
    body_fn_inputs = [
        carried_inputs[0],
    ] + fake_carried_inputs[1:] + list(additional_inputs)
    cond_fn_inputs = carried_inputs + additional_inputs
  else:
    body_fn_inputs = carried_inputs
    cond_fn_inputs = carried_inputs

  # due to `xla::While` requirement, body xlacomputation inputs/outputs, cond xlacomputation and init need to be the same shape and type;
  # and carried_inputs contain (iter, values), additional_inputs contain (weights/bias)
  # based on generated body xlacomputation outputs: (iter, weights/bias, values)
  # we create expected order for cond/body xlacomputation generation to compare and match: (iter, weights/bias, values)
  dummy_inputs_list = [
      fake_carried_inputs[0],
  ] + fake_additiona_args + fake_carried_inputs[1:]

  #  ====== body_fn ======
  body_result = body_fn(*body_fn_inputs)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")

  #  ====== body xlacomputation ======
  body_ctx.buildforiloop(list(body_result), dummy_inputs_list)
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)

  #  ====== cond_fn ======
  cond_result = cond_fn(*cond_fn_inputs)
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
  total_inputs = tuple([
      iter_value,
  ]) + tuple(additional_inputs) + tuple(input_and_outputs_value)

  kwargs = {}
  if type(total_inputs) is tuple:
    shapes = xb.tensor_shape(total_inputs)
  else:
    shapes = xb.tensor_shape((total_inputs))
  builder = xb.create_builder('while_loop')
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

  # unwrapper result without additional_inputs for original order
  additional_inputs_len = len(additional_inputs) + 1
  final_res = [
      result[0],
  ] + result[additional_inputs_len:]

  return final_res
