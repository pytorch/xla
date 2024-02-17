# to run, current need to patch this file's code to end of PyTorch/torch/_higher_order_ops/while_loop.py
import numpy as np
import torch
import torch_xla
# from torch.library import Library, impl
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor
# import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
import torch._higher_order_ops.while_loop # make sure they implement this line code: `while_loop_op = HigherOrderOperator("while_loop")`
from torch._higher_order_ops.while_loop import while_loop_op
# while_loop_op = HigherOrderOperator("while_loop")

# @while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, operands):
  # cond_fn&body_fn: callable
  # operands: (Tuple of possibly nested dict/list/tuple of tensors)
  return _xla_while_loop(cond_fn, body_fn, operands)

def _xla_while_loop(cond_fn, body_fn, operands):
  def op_fn(internal_x):
    zero = xb.Op.scalar(internal_x.builder(), 0, dtype=xb.Type.S32)
    w = xb.Op.mkwhile((zero, internal_x), cond_fn, body_fn)
    return w.get_tuple_element(1)
  op = xor.register('test_while', op_fn)
  return xu.as_list(op(operands[0]))