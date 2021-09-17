import torch
import torch_xla
import torch_xla.core.xla_model as xm
import logging
import functools

debug = False
BYPASS_XLA = False  #Performance profiling experimentation

class DynamicTensorXLASize(object):
    def __init__(self, s, t=None):
        self.static_size = s
        self.dynamic_size = t

class XLATensor(torch.Tensor):
    def __new__(cls, data, **kwargs):
        return torch.Tensor._make_subclass(cls, torch.as_tensor(data, dtype=torch.float32, **kwargs))

    def __init__(self, data, **kwargs):
        super().__init__()
        if debug: print('[__init__]')
        self.t_ = torch.as_tensor(data, dtype=torch.float32, **kwargs)

    def size(self):
        if debug: print('[size] ')
        static_size = super(XLATensor, self).size()
        if self.device == xm.xla_device():
            dynamic_size = torch_xla._XLAC._get_xla_tensor_dimension_size(self.t_, 0)
            return DynamicTensorXLASize(static_size, self.t_) #TODO: replace self.t_ with dynamic_size
        else:
            return DynamicTensorXLASize(static_size)

    def expand(self, size):
        if debug: print('[expand]')
        if self.device == xm.xla_device():
            if size.dynamic_size != None:
                if BYPASS_XLA == True:
                    result = self.t_.expand(size.static_size) 
                else:
                    result = torch_xla._XLAC._xla_dynamic_expand(self.t_, size.dynamic_size, size.static_size)
                return XLATensor(result.tolist()) #Q: return XLATensor or torch.Tensor?
        return self.t_.expand(size.static_size)

    def __repr__(self):
        if debug: print ('[__repr__]')
        return "XLATensor:\n{}\n".format(self.t_)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if debug: print ('[__torch_function__] ', func.__name__)
        if func is not torch.Tensor.__repr__:
            logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        args = [a.t_ if hasattr(a, 't_') else a for a in args]
        return func(*args, **kwargs)

if debug: print ('[main] make b')
t = XLATensor([[1], [2], [3]], device=xm.xla_device())
print('Tensor t:\n', t)

if debug: print ('[main] make o')
o = XLATensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]], device=xm.xla_device())
print('Tensor o:\n', o)

if debug: print ('[main] size o')
b = o.size()
print('Size output:\n\tstatic_size = %s\n\tdynamic_size = %s\n' % (b.static_size, b.dynamic_size))

if debug: print ('[main] expand t')
tt = t.expand(b)
print ('Expand output:\n', tt)

if debug: print ('[main] add')
print("Add output:\n", torch.add(tt, o))
