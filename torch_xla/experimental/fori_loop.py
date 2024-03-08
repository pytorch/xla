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


def fori_loop(lower, upper, body_fun, init_val): # *init_val):
  # def new_cond(pred, token, x):
  #   return pred
  # token, pred = cond(token, x)
  # while new_cond(pred, token, x):
  #   token, x = body(token, x)
  #   token, pred = cond(token, x)


  # example data:
  # init_val = torch.tensor([0], dtype=torch.int32, device=device)
  # lower = torch.tensor([0], dtype=torch.int32, device=device)
  # upper = torch.tensor([10], dtype=torch.int32, device=device)
  # limit_range = upper - lower
  device = xm.xla_device()
  # one_value = torch.tensor([0], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  # lower, upper
  limit_value = upper
  init = lower
  iterator = lower

  # device = xm.xla_device()

  ### might be current JAX logic:
  # # def new_cond(pred, token, x):
  # #   return pred
  # # token, pred = cond(token, x)
  # # while new_cond(pred, token, x):
  # #   token, x = body(token, x)
  # #   token, pred = cond(token, x)

  # # fori_loop(lower, upper, body_fun, *init_val):
  # def new_cond(pred, iterator, init_val): # cond_fn
  #   pred = (iterator <= limit_value)
  #   return pred
  
  # def cond(iterator, init_val):
  #   return iterator, init_val
  
  # def body_fn(pred, iterator, init_val): # (pred, token, init_val):
  #   iterator, x = body_fun(iterator, init_val) # token, x = body(token, x)
  #   iterator, pred = cond(iterator, init_val) # token, pred = cond(token, x)
  #   return pred, iterator, init_val

  ### earlist fori_loop logic:
  # def fori_loop(lower, upper, body_fun, init_val):
  # """Loop from `lower` to `upper` by reduction to `while_loop`.

  # Arguments:
  #   lower: loop index lower bound (inclusive)
  #   upper: loop index upper bound (exclusive)
  #   body_fun: function of type (int, T) -> T, where T is the type of `init_val`
  #   init_val: initial loop value, of type T

  # Returns:
  #   Loop value from the final iteration, of type T.
  # """
  # # state: (upper limit, index, loop value)
  # # The `lt` and `add` functions are added to the namespace programmatically.
  # _, _, result = _while_loop(
  #     lambda (upper, i, _): lt(i, upper),
  #     lambda (upper, i, x): (upper, add(i, 1), body_fun(i, x)),
  #     (upper, lower, init_val))
  # return result

  # def cond_fun(upper, i, _):
  #   return i <= upper # lt(i, upper)

  # def body_fun(upper, i, x):
  #   return (upper, add(i, 1), body_fun(i, x))

  # init_val:
  #   (upper, lower, init_val)

  ### earlist while_loop logic:
  #def _while_loop(cond_fun, body_fun, init_val):
  # init_val = recursive_pack(init_val)
  # # TODO(mattjj,dougalm): raise to Shaped, should error if higher than shaped.
  # # (should also check that output is at least as low as input.)
  # pval = _abstractify(init_val)
  # body_fun2 = lambda x: recursive_pack(body_fun(x))
  # cond_jaxpr, _, cond_consts = pe.trace_unwrapped_to_jaxpr(cond_fun, (pval,))
  # body_jaxpr, pvout, body_consts = pe.trace_unwrapped_to_jaxpr(body_fun2, (pval,))
  # abs_out, _ = pvout
  # return while_p.bind(init_val, abs_out=abs_out,
  #                     cond_jaxpr=cond_jaxpr, cond_consts=cond_consts,
  #                     body_jaxpr=body_jaxpr, body_consts=body_consts)

  #def while_loop_translation_rule(
  #   c, init_val, abs_out, cond_jaxpr, cond_consts, body_jaxpr, body_consts):
  # shape = c.GetShape(init_val)
  # cond_computation = xla.jaxpr_computation(cond_jaxpr, cond_consts, (), shape)
  # body_computation = xla.jaxpr_computation(body_jaxpr, body_consts, (), shape)
  # return c.While(cond_computation, body_computation, init_val)


  #def recursive_pack(x):
  # if type(x) is tuple:
  #   return core.pack(map(recursive_pack, x))
  # else:
  #   return x

  #def _abstractify(x):
  # # abstractify wrapper used internally for primitives like _while_loop
  # if isinstance(x, core.Tracer):
  #   # TODO(mattjj): check that it's at least ShapedArray
  #   return pe.PartialVal((x.aval, core.unit))
  # else:
  #   return pe.PartialVal((xla.abstractify(x), core.unit))


  # def cond_fun(upper, i, _):
  #   return i <= upper # lt(i, upper)

  # def body_fun(upper, i, x):
  #   return (upper, add(i, 1), body_fun(i, x))

  # init_val:
  #   (upper, lower, init_val)
  def cond_fn(upper, lower, init_val): # (upper, i, _):
    return lower <= upper # i <= upper # lt(i, upper)
    # one_value = torch.tensor([1], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
    # one_value_2 = torch.tensor([1], dtype=torch.int32, device=device)
    # # torch.add(one_value, one_value_2)
    # a = torch.add(init_val[0], one_value[0])
    # b = torch.add(init_val[0], one_value_2[0])
    # # c = (a[0] >= b[0]) and init_val[0] and init_val[1] # (limit_value[0] <= init[0])
    # # c = (a[0] >= b[0]) and init_val[0] and init_val[1]
    # return (limit_value[0] <= init[0]) and init_val[0] and init_val[1]

  def body_fn(upper, lower, init_val): # (upper, i, x):
    return (upper, torch.add(lower, 1), body_fun(lower, init_val)) # (upper, add(i, 1), body_fun(i, x))
    # one_value = torch.ones(1, dtype=torch.int32, device=device)
    # return (torch.add(init, one_value), limit_value.clone(), body_fun(*init_val), init_val[1])

    # TODO(@manfei): init and limit_value has to be torch.tensor.
    # init = torch.tensor([0], dtype=torch.int32, device=device)
    # limit_value = torch.tensor([10], dtype=torch.int32, device=device)
  res = while_loop(cond_fn, body_fn, (upper, lower, init_val)) # (init, limit_value, *init_val))
  # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
  # self.assertEqual(expected, res)
  return res

  # def cond_fn(init, limit_value, *init_val):
  #   one_value = torch.tensor([1], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  #   one_value_2 = torch.tensor([1], dtype=torch.int32, device=device)
  #   # torch.add(one_value, one_value_2)
  #   a = torch.add(init_val[0], one_value[0])
  #   b = torch.add(init_val[0], one_value_2[0])
  #   # c = (a[0] >= b[0]) and init_val[0] and init_val[1] # (limit_value[0] <= init[0])
  #   # c = (a[0] >= b[0]) and init_val[0] and init_val[1]
  #   return (limit_value[0] <= init[0]) and init_val[0] and init_val[1]

  # def body_fn(init, limit_value, *init_val):
  #   one_value = torch.ones(1, dtype=torch.int32, device=device)
  #   return (torch.add(init, one_value), limit_value.clone(), body_fun(*init_val), init_val[1])

  #   # TODO(@manfei): init and limit_value has to be torch.tensor.
  #   # init = torch.tensor([0], dtype=torch.int32, device=device)
  #   # limit_value = torch.tensor([10], dtype=torch.int32, device=device)
  # res = while_loop(cond_fn, body_fn, (init, limit_value, *init_val))
  # # expected = _fake_while_loop(cond_fn, body_fn, (init, limit_value))
  # # self.assertEqual(expected, res)
  # return res

  # def cond_fn(lower, upper, *init_val):
  #   one_value = torch.tensor([1], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  #   one_value_2 = torch.tensor([0], dtype=torch.int32, device=device)
  #   lower = torch.add(lower, one_value)
  #   lower = torch.add(lower, one_value_2)
  #   # init_val = init_val.clone()
  #   # print("lower: ",lower)
  #   return lower[0] <= upper[0]
  
  # def body_fn(lower, upper, *init_val):
  #   # one_value = torch.tensor([0], dtype=torch.int32, device=device) # torch.ones(1, dtype=torch.int32, device=device)
  #   # lower = torch.add(lower, one_value)
  #   init_val_local = body_fun(*init_val)
  #   return (lower.clone(), upper.clone(), init_val_local, init_val[1])

  # a = while_loop(cond_fn, body_fn, (lower, upper, *init_val))
  # # print("result: finall: ", a)
  # return a # while_loop(cond_fn, body_fn, (lower, upper, init_val))


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
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # generate cond_fn xlacomputation
  cond_result = cond_fn(*operands)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  cond_ctx.build([cond_result])
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)

  # generate body_fn xlacomputation
  body_result = body_fn(*operands)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.build(list(body_result))
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)

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
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 tuple(operands), computation)

  return result
