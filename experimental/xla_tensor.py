import torch
import torch_xla
import torch_xla.core.xla_model as xm
import logging
import functools
import time
from collections import namedtuple

debug = False
SIMPLE_TEST = False

class XLATensor(torch.Tensor):
    def __new__(cls, data, **kwargs):
        return torch.Tensor._make_subclass(cls, torch.as_tensor(data, dtype=torch.float32, **kwargs))

    def __init__(self, data, **kwargs):
        super().__init__()
        # if debug: print('[__init__]')
        self.t_ = torch.as_tensor(data, dtype=torch.float32, **kwargs)
        self.xla_tensor_sizes_ = namedtuple("XLATensorSizes", "static_size dynamic_size")

    def size(self):
        # if debug: print('[size] ')
        static_size = super(XLATensor, self).size()
        dynamic_size = torch_xla._XLAC._get_xla_tensor_dimension_size(self.t_, 0)
        xla_tensor_sizes = self.xla_tensor_sizes_(static_size, self.t_) #TODO: replace self.t_ with dynamic_size
        return xla_tensor_sizes 
        #DynamicTensorXLASize(static_size) #TODO: done when no dynamic shame

    def expand(self, size):
        # if debug: print('[expand]')
        # if size.dynamic_size != None:
        # result = self.t_.expand(size.static_size) 
        result = torch_xla._XLAC._xla_dynamic_expand(self.t_, size.dynamic_size, size.static_size)
        return XLATensor(result.tolist()) #Q: return XLATensor or torch.Tensor?
        # return self.t_.expand(size.static_size)

    def __repr__(self):
        # if debug: print ('[__repr__]')
        return "XLATensor:\n{}\n".format(self.t_)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # if debug: print ('[__torch_function__] ', func.__name__)
        # if func is not torch.Tensor.__repr__:
            # logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        args = [a.t_ if hasattr(a, 't_') else a for a in args]
        return func(*args, **kwargs)

if SIMPLE_TEST == True:
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
else:
    NUM = 50

    time_size = 0
    time_expand = 0
    for i in range(NUM):
        t = torch.tensor([[1], [2], [3]], device=xm.xla_device())
        o = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]], device=xm.xla_device())

        tic = time.perf_counter()
        b = o.size()
        toc = time.perf_counter()
        time_size += toc-tic

        tic = time.perf_counter()
        tt = t.expand(b)
        toc = time.perf_counter()
        time_expand += toc-tic
    print(f"size() time {time_size:0.4f} seconds")
    print(f"expand() time {time_expand:0.4f} seconds")

    time_size = 0
    time_expand = 0
    for i in range(NUM):
        t = XLATensor([[1], [2], [3]], device=xm.xla_device())
        o = XLATensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]], device=xm.xla_device())

        tic = time.perf_counter()
        b = o.size()
        toc = time.perf_counter()
        time_size += toc-tic

        tic = time.perf_counter()
        tt = t.expand(b)
        toc = time.perf_counter()
        time_expand += toc-tic
    print(f"size() time {time_size:0.4f} seconds")
    print(f"expand() time {time_expand:0.4f} seconds")