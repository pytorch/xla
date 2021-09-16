import torch
import torch_xla
import torch_xla.core.xla_model as xm
import logging
import functools

debug = False

class DynamicTensorXLASize(object):
    def __init__(self, s, t=None):
        self.static_size = s
        self.dynamic_size_tensor = t

class XLATensor(torch.Tensor):
    def __new__(cls, data, **kwargs):
        return torch.Tensor._make_subclass(cls, torch.as_tensor(data, dtype=torch.float32, **kwargs))

    def __init__(self, data, **kwargs):
        super().__init__()
        if debug: print('[__init__]')
        self.dev_ = 'TPU' #Q: how to replace this with xm.xla_device()?
        self.t_ = torch.as_tensor(data, dtype=torch.float32, **kwargs)# Q: replace with parent class tenor obj?

    def size(self):
        if debug: print('[size] ')
        siz = super(XLATensor, self).size()
        if self.dev_ == 'TPU':
            return DynamicTensorXLASize(siz, self)
            #return torch_xla.dynamic_size(self.tensor) #TBD
        else:
            return DynamicTensorXLASize(siz)

    def expand(self, size):
        if debug: print('[expand]')
        if self.dev_ == 'TPU':
            if size.dynamic_size_tensor != None:
                result = self.t_.expand(size.static_size) 
                #result = torch_xla.dynamic_expand(size) #TBD
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
print('Size output:\n\tstatic_size = %s\n\tdynamic_size_tensor =\n\t%s' % (b.static_size, b.dynamic_size_tensor))

if debug: print ('[main] expand t')
tt = t.expand(b)
print ('Expand output:\n', tt)

if debug: print ('[main] add')
print("Add output:\n", torch.add(tt, o))
