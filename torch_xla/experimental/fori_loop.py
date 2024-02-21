import numpy as np
import torch
import torch_xla
from torch.library import Library, impl
import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor
import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator

while_loop_op = HigherOrderOperator("while_loop")

@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  print("arrive the xla_while_loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  return _xla_while_loop(cond_fn, body_fn, operands)

def _xla_while_loop(cond_fn, body_fn, operands):
  # operands: list[Tensor]
  print("arrive the xla_while_loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("operands type: ", type(operands))
  # internal_operands = operands[0]  # specific to test case operands: (x, )
  print("before define the op_fn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  def op_fn(internal_x):
    # def cond(internal_x):
    #   return cond_fn(internal_x)
    # def body(internal_x):
    #   return body_fn(internal_x)
    zero = xb.Op.scalar(internal_x.builder(), 0, dtype=xb.Type.S32)
    w = xb.Op.mkwhile((zero, internal_x), cond_fn, body_fn)
    return w.get_tuple_element(1)
  print("after define the op_fn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  op = xor.register('test_while', op_fn)
  return xu.as_list(op(operands[0]))