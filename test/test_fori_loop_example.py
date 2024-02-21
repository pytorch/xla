import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor

# import pdb; pdb.set_trace()

# def test_while(self):

#     def op_fn(a, b, limit=None):

def op_fn(a, b, limit=10): #None):
  def cond(counter, a, b):
    print("cond a", a)
    print("cond b", b)
    print("cond counter", counter)
    print("cond builder", counter.builder())
    return counter < xb.Op.scalar(counter.builder(), limit, dtype=xb.Type.S32)

  def body(counter, a, b):
    print("body a", a)
    print("body b", b)
    print("body counter", counter)
    next_counter = counter + xb.Op.scalar(counter.builder(), 1, dtype=xb.Type.S32)
    return xb.Op.tuple((next_counter, a + b, b))

  print("after two func define")
  print("a: ", a)
  print("b: ", b)
  print("a.builder: ", a.builder())
  # print("a.builder.name: ", a.builder().name())
  zero = xb.Op.scalar(a.builder(), 0, dtype=xb.Type.S32)
  print("zero: ", zero)
  w = xb.Op.mkwhile((zero, a, b), cond, body)
  print("w: ", w)
  return w.get_tuple_element(1)


# def cond(counter, a, b):
#   return counter < xb.Op.scalar(counter.builder(), limit, dtype=torch.int32)

# def body(counter, a, b):
#   next_counter = counter + xb.Op.scalar(counter.builder(), 1, dtype=torch.int32)
#   return xb.Op.tuple((next_counter, a + b, b))

op = xor.register('test_while', op_fn)

device = xm.xla_device()
a = torch.randn(2, 2) # .to(device)
b = torch.randn(2, 2) # .to(device)
test_tensors = [a, b]
test_tensors = xu.as_list(test_tensors)
xla_tensors = [x.to(device).detach().requires_grad_(x.requires_grad) for x in test_tensors]

# kwargs=dict()
# results = xu.as_list(aten_fn(*tensors, **kwargs))
xla_a = a.to(device)
xla_b = b.to(device)

ccc = op(xla_a, xla_b)
xla_try_result = xu.as_list(ccc) #, kwargs={'limit': 10}) # op(xla_a, xla_b))
# xla_results = xu.as_list(op(*xla_tensors)) # , **kwargs))

print("xla_try_result: ", xla_try_result)
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