import torch
import torch_xla
import torch_xla.core.xla_model as xm
import logging
import functools
import time
from collections import namedtuple

debug = False
SIMPLE_TEST = True
time_torch_function = 0

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
        static_size = self.t_.size()
        dynamic_size = []
        for i,_ in enumerate(self.t_.shape):
            dynamic_size.append(torch_xla._XLAC._get_xla_tensor_dimension_size(self.t_, i))
        return self.xla_tensor_sizes_(static_size, dynamic_size[0])

    def expand(self, size):
        # if debug: print('[expand]')
        return torch_xla._XLAC._xla_dynamic_expand(self.t_, size.dynamic_size, size.dynamic_size.int(), size.static_size)

    def nonzero(self):
        return torch.nonzero(self.t_)

    def __repr__(self):
        # if debug: print ('[__repr__]')
        return "XLATensor:\n{}\n".format(self.t_)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a.t_ if hasattr(a, 't_') else a for a in args]
        result = func(*args, **kwargs)
        return result

if SIMPLE_TEST == True:
    a = XLATensor([[0, 0, 0], [1, 1, 0]], device=xm.xla_device())
    b = XLATensor([[1], [2]], device=xm.xla_device())

    idxs = a.nonzero()
    idxs_t = XLATensor([[1]], device=xm.xla_device())
    idxs_t.t_ = idxs
    print('a.nonzero(): ', idxs)

    y = b.expand(idxs_t.size())
    print (f"b.expand(): {y}")
    y_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(y, 0)
    print("[RUNNING] _get_xla_tensor_dimension_size(y, 0)\n", y_dim0_shape)
    print("[RUNNING] _get_xla_tensors_hlo([y])\n", torch_xla._XLAC._get_xla_tensors_hlo([y]))
    print("[RUNNING] _get_xla_tensors_text([y])\n", torch_xla._XLAC._get_xla_tensors_text([y]))
else:
    NUM = 500

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
    print(f"torch_function() time {time_torch_function:0.4f} seconds")