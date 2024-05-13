
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
  return _xla_while_loop(cond_fn, body_fn, carried_inputs, additional_inputs)
  # return _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs)


# ----------------- PyLoweringContext --------------------------------
# MNIST's _xla_while_loop's pre func to summary args
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
  return _xla_while_loop_target(cond_fn, new_body_fn, carried_inputs, additional_inputs)

# MNIST's _xla_while_loop with PyLoweringContext
def _xla_while_loop_target(cond_fn, body_fn, carried_inputs, additional_inputs=None):
  output_value_index = len(carried_inputs) - 1
  # print("arrive _xla_while_loop")
  # print("carried_inputs: ", carried_inputs)
  # print("type carried_inputs: ", type(carried_inputs))
  # print("additional_inputs: ", additio