import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor
import threading
import pickle

# import pdb; pdb.set_trace()
# def test_while(self):
#     def op_fn(a, b, limit=None):

# ##############get HLO of the body func
# >>> device = xm.xla_device()
# >>> a = torch.rand(10, device=device)
# >>> b = torch.rand(10, device=device)
# >>> xm.mark_step()
# >>> result = torch.add(a, b)
# >>> ctx = torch_xla._XLAC.lowering.LoweringContext()
# >>> ctx.build([result])
# >>> hlo = ctx.hlo()
# >>> hlo_text = ctx.hlo_text()
# ##############

# ##############get xlacomputation from hlo
# # >>> import torch_xla.core.xla_builder as xb
# >>> axlacomputation = xb.computation_from_module_proto("testxlacompujan261248am", hlo)
# >>> aaahlotext = xb.get_computation_hlo(axlacomputation)
# # >>> axlacomputation
# # <_XLAC.XlaComputation object at 0x7feeea0c9ff0>
# # >>> aaahlotext
# # 'HloModule PyLoweringContext.11, entry_computation_layout={(f32[10]{0}, f32[10]{0})->(f32[10]{0})}\n\nENTRY %PyLoweringContext.11 (p0.6: f32[10], p1.8: f32[10]) -> (f32[10]) {\n  %p1.8 = f32[10]{0} parameter(1)\n  %p0.6 = f32[10]{0} parameter(0)\n  %constant.1 = f32[] constant(1)\n  %reshape.2 = f32[1]{0} reshape(f32[] %constant.1)\n  %broadcast.3 = f32[1]{0} broadcast(f32[1]{0} %reshape.2), dimensions={0}\n  %reshape.4 = f32[] reshape(f32[1]{0} %broadcast.3)\n  %broadcast.5 = f32[10]{0} broadcast(f32[] %reshape.4), dimensions={}\n  %multiply.7 = f32[10]{0} multiply(f32[10]{0} %p0.6, f32[10]{0} %broadcast.5)\n  %add.9 = f32[10]{0} add(f32[10]{0} %p1.8, f32[10]{0} %multiply.7)\n  ROOT %tuple.10 = (f32[10]{0}) tuple(f32[10]{0} %add.9)\n}\n\n'
# # >>> 
# ##############

device = xm.xla_device()
cpu_a = torch.randn(2, 2) # .to(device)
cpu_b = torch.randn(2, 2) # .to(device)
a = cpu_a.to(device)
b = cpu_b.to(device)
# xla_a = a.to(device)
# xla_b = b.to(device)
counter = torch.ones(1).to(device) # torch.int(1)
inputs = [a, b]
name = 'fori_loop_ed_torch_func' # 'testforiloophere'
opname = 'xla::_op_' + name

# define body function
def body(torch_func, counter, a, b):
    # torch_func: python code;
    # counter: (should be) python code here
    # a: python code, input a
    # b: python code, input b
    # next_counter = counter + xb.Op.scalar(counter.builder(), 1, dtype=xb.Type.S32)
  next_counter = counter + 1
  return (next_counter, torch_func(a, b), b)

# get body HLO
xm.mark_step()
body_result = body(torch.add, counter, a. b) # torch.add(a, b)
body_ctx = torch_xla._XLAC.lowering.LoweringContext()
body_ctx.build([body_result])
body_hlo = body_ctx.hlo()
# body_hlo_text = body_ctx.hlo_text()

# get body XLAcomputation from body HLO
body_computation = xb.computation_from_module_proto("bodycomputation", body_hlo)

# def fori_loop_op_fn(torch_func, a, b, limit=10): #None):
def cond(counter, a, b):
  return counter < limit
#   return counter < xb.Op.scalar(counter.builder(), limit, dtype=xb.Type.S32)

# get cond HLO
xm.mark_step() # clean formal irs
cond_result = cond(counter, a. b)
# ctx = torch_xla._XLAC.lowering.LoweringContext()
cond_ctx.build([cond_result])
cond_hlo = cond_ctx.hlo()
# cond_hlo_text = cond_ctx.hlo_text()

# get cond XLAcomputation from cond HLO
cond_computation = xb.computation_from_module_proto("condcomputation", cond_hlo)

# def fori_loop_op_fn(torch_func, a, b, limit=10): #None):
#   def cond(counter, a, b):
#     return counter < xb.Op.scalar(counter.builder(), limit, dtype=xb.Type.S32)

#   def body(torch_func, counter, a, b):
#     # torch_func: python code;
#     # counter: (should be) python code here
#     # a: python code, input a
#     # b: python code, input b
#     # next_counter = counter + xb.Op.scalar(counter.builder(), 1, dtype=xb.Type.S32)
#     next_counter = counter + 1
#     return (next_counter, torch_func(a, b), b)
#     # return xb.Op.tuple((next_counter, a + b, b))

  ###get body_computation from body
  

#   zero = xb.Op.scalar(a.builder(), 0, dtype=xb.Type.S32) # counter init value
#   # zero = 0
#   w = xb.Op.mkwhile((zero, a, b), cond, body_computation)
#   #w = xb.Op.mkwhile((zero, a, b), cond, body)

def fori_loop_internal_defination_op_fn(zero, a, b, cond_computation, body_computation): #None):
  w = xb.Op.mkwhile(xb.Op.tuples(zero, a, b), cond_computation, body_computation)
  # p = input_tuple.while_loop(
  return w.get_tuple_element(1)

# op = xor.register('test_while', op_fn)
################register new op
#   def __call__(self, *args, **kwargs):
    # shapes = xb.tensor_shape(args[1:]) #args[1:] == [a, b]
kwargs = {}
# shapes = xb.tensor_shape(xb.Op.tuples(a, b))
shapes = xb.tensor_shape(xb.Op.tuples(inputs))
key = pickle.dumps([shapes, kwargs])
with threading.Lock(): # self._lock:
  computation = self._computations.get(key, None)
  if computation is None:
#TOD0: repalce with xb.create_computation mentioned below
#        computation = xb.create_computation(self._name, self._opfn, shapes,
#                                            **kwargs)
    computation = xb.create_computation(name, fori_loop_internal_defination_op_fn,
                                        shapes, **kwargs)
    self._computations[key] = computation
    # if xu.getenv_as('XLA_OP_PRINT_COMPUTATIONS', bool, False):
    #   print(xb.get_computation_hlo(computation), file=sys.stderr)
  result = torch_xla._XLAC._xla_user_computation(opname, inputs,
                                                 computation)
#   result = torch_xla._XLAC._xla_user_computation(opname, args,
#                                                  computation)
return result[0] if len(result) == 1 else result
################

################return result --- above has laready returned
# ccc = op(xla_a, xla_b) # this execute like above return
# xla_try_result = xu.as_list(ccc) #, kwargs={'limit': 10}) # op(xla_a, xla_b))
# xla_results = xu.as_list(op(*xla_tensors)) # , **kwargs))

# print("xla_try_result: ", xla_try_result)

# ################xb.create_computation
# # def create_computation(name, fn, shapes, **kwargs):
#   builder = create_builder(name)
#   params = []
#   for shape in shapes:
#     p = xb.mkparam(builder, len(params), shape)
#     params.append(p)

#this part is to generate computation
#   root = fn(*params, **kwargs)
#   return root.build(name)
################

# device = xm.xla_device()
# a = torch.randn(2, 2) # .to(device)
# b = torch.randn(2, 2) # .to(device)
# test_tensors = [a, b]
# test_tensors = xu.as_list(test_tensors)
# xla_tensors = [x.to(device).detach().requires_grad_(x.requires_grad) for x in test_tensors]

# kwargs=dict()
# results = xu.as_list(aten_fn(*tensors, **kwargs))
# xla_a = a.to(device)
# xla_b = b.to(device)

# ccc = op(xla_a, xla_b)
# xla_try_result = xu.as_list(ccc) #, kwargs={'limit': 10}) # op(xla_a, xla_b))
# # xla_results = xu.as_list(op(*xla_tensors)) # , **kwargs))

# print("xla_try_result: ", xla_try_result)
# zero = xb.Op.scalar(a.builder(), 0, dtype=torch.int32, device = device)
# zero = torch.zeros(1).to(device)
# zero = (torch.zeros(4, 3, 224, 224), )
# zero = xb.Op.scalar()
# print("a: ", a)
# print("a.builder", a.builder)

# print("zero", zero)
# w = xb.Op.mkwhile((zero, a, b), cond, body)
# print("w", w)
# print("return value", w.get_tuple_element(1))

#    def aten_fn(a, b, limit=None):
#      for _ in range(0, limit):
#        a = a + b
#      return a

#    self.runOpBuilderTest(
#        'test_while', [torch.randn(2, 2), torch.randn(2, 2)],
#        op_fn,
#        aten_fn=aten_fn,
#        kwargs={'limit': 10})
