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

  # create inputs placeholder
  kwargs = {}
  shapes = xb.tensor_shape(operands)
  builder = xb.create_builder('test_while')
  params = []
  # secondparams = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)
    # secondparams.append(xb.Op.tuple([p]))

  # generate cond_fn xlacomputation
  xm.mark_step()
  cond_result = cond_fn(operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.setnamestring("condctx")
  cond_ctx.build(list(cond_result))
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)

  # generate body_fn xlacomputation
  xm.mark_step()
  body_result = body_fn(operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.setnamestring("bodyctx")
  body_ctx.build(list(body_result))
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)

  # create xla:While op with cond_computation and body_computation
  input_tuple = xb.Op.tuple(params)
  aaa_tuple = xb.Op.get_tuple_element(input_tuple, 0)
  w = xb.mkop(
      'While', [aaa_tuple.op],
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)

  # operands would be changed from torch.tensor([1]) to torch.tensor(1) after torch.compile when call torch._higher_order_ops.while_loop, so create a new input tesor here
  localoperands = torch.tensor([1], dtype=torch.int32, device=xm.xla_device())
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (localoperands,), computation)

  return result
