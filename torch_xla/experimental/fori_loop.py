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

  def op_fn(internal_x):
    # TODO(manfei): replace cond_fn_placeholder and body_fn_placeholder after lower fn to xlacomputation and could be in xla::while
    def cond_fn_placeholder(counter, internal_x):
      return counter < xb.Op.scalar(internal_x.builder(), 10, dtype=xb.Type.S32)

    def body_fn_placeholder(counter, internal_x):
      next_counter = counter + xb.Op.scalar(
          counter.builder(), 1, dtype=xb.Type.S32)
      internal_x = internal_x + xb.Op.scalar(
          internal_x.builder(), 1, dtype=xb.Type.S32)
      return xb.Op.tuple((next_counter, internal_x))

    zero = xb.Op.scalar(internal_x.builder(), 0, dtype=xb.Type.S32)
    w = xb.Op.mkwhile((zero, internal_x), cond_fn_placeholder,
                      body_fn_placeholder)
    return w.get_tuple_element(1)

  op = xor.register('test_while', op_fn)
  return xu.as_list(op(operands[0]))
