from typing import Any, Callable, ClassVar, Dict, Iterable, Tuple
import torch

from torch._ops import OpOverload

from _coo_ops import (
    coo_sparse_mask,
    coo_resize_as_,
    coo_coalesce,
    coo_add_,
    coo_indices,
    coo_values,
    coo_nnz,
)

__all__ = ["SparseCOOTensor"]

aten = torch.ops.aten


class SparseCOOTensor(torch.Tensor):

  DISPATCH_TABLE: ClassVar[Dict[OpOverload, Callable]] = {}
  _v: torch.Tensor
  _i: torch.Tensor

  def __new__(cls,
              indices: torch.Tensor,
              values: torch.Tensor,
              size: Iterable[int],
              *,
              dtype: torch.dtype | None = None,
              device: torch.device | None = None,
              requires_grad: bool | None = None):
    device = device if device is not None else values.device
    dtype = dtype if dtype is not None else values.dtype
    requires_grad = requires_grad if requires_grad is not None else values.requires_grad
    assert (device == values.device and device == indices.device and
            dtype == values.dtype and requires_grad == values.requires_grad)

    res = torch.Tensor._make_wrapper_subclass(
        cls,
        size=size,
        strides=None,
        layout=torch.sparse_coo,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad)
    res._v = values
    res._i = indices
    return res

  @classmethod
  def __torch_dispatch__(cls,
                         func: OpOverload,
                         types: Tuple,
                         args: Tuple = (),
                         kwargs: Dict[str, Any] | None = None) -> torch.Tensor:
    impl = cls._match_func_key_for_dispatch(func)
    return impl(args, kwargs)

  @classmethod
  def _load_sparse_dispatch(cls):
    if getattr(cls, "DISPATCH_TABLE", None) is None:
      cls.DISPATCH_TABLE = {
          aten.values: coo_values,
          aten.indices: coo_indices,
          aten.coalesce: coo_coalesce,
          aten.add_: coo_add_,
          aten.sparse_mask: coo_sparse_mask,
          aten.resize_as_: coo_resize_as_,
          aten.nnz: coo_nnz,
      }

  @classmethod
  def _match_func_key_for_dispatch(cls, func):
    cls.load_sparse_dispatch()
    impl = cls.DISPATCH_TABLE.get(func, None)
    if impl is None:
      impl = cls.DISPATCH_TABLE.get(func._overloadpacket,
                                    cls._make_not_implemented(func))
    return impl

  @classmethod
  def _make_not_implemented(cls, func):

    def impl(args=(), kwargs=None):
      raise NotImplemented(
          f"Sparse support via {cls.__name__} is limited to a very narrow "
          f"scope. The {func.__name__} operator is not currently supported. It"
          " is likely you are using this outside of its intended purpose "
          "if you see this message. Supported operators are: ",
          f"{list(k.__name__ for k in cls.DISPATCH_TABLE.keys())}")

    return impl
