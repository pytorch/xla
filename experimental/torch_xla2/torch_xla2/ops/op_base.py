import torch
from torch_xla2 import interop


class BinaryOpWithPromotion:

    def __init__(self, inner):
        self.inner = inner 

    def _get_dtype(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.dtype
        else:
            if isinstance(obj, float):
                return torch.float32
            if isinstance(obj, int):
                return torch.int32
        return torch.float32


    def __call__(self, *args, **kwargs):
        # args are torch.Tensor
        res = interop.torch_view(self.jax)(*args, **kwargs)

        dtype = torch.promote_types(
            self._get_dtype(args[0]), 
            self._get_dtype(args[1]))
        if dtype != res.dtype:
            res = res.to(dtype)
        return res


class InplaceOp:

    def __init__(self, functional_op, position_to_mutate=0):
        self.functional = functional_op
        self.position_to_mutate = position_to_mutate

    def __call__(self, *args, **kwargs):
        to_mutate = args[0]
        to_mutate.copy_(self.functional(*args, **kwargs))
        return to_mutate


class OutVariant:

    def __call__(self, *args, **kwargs):
        to_mutate = kwargs['out']
        del kwargs['out']
        to_mutate._elem = self.functional(*args, **kwargs)._elem
        return to_mutate




