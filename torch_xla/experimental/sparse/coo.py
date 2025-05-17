from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Tuple
import warnings
import torch

from torch._ops import OpOverload
from torch.overrides import enable_reentrant_dispatch, get_default_nowrap_functions

__all__ = ["SparseCOOTensor"]

aten = torch.ops.aten


class SparseCOOTensor(torch.Tensor):

  # DISPATCH_TABLE: ClassVar[Dict[OpOverload, Callable]] = {
  #     aten.values: coo_values,
  #     aten.indices: coo_indices,
  #     aten._indices: coo_indices_unsafe,
  #     aten._values: coo_values_unsafe,
  #     aten.coalesce: coo_coalesce,
  #     aten.add_: coo_add_,
  #     aten.sparse_mask: coo_sparse_mask,
  #     aten.resize_as_: coo_resize_as_,
  #     aten.clone: coo_clone,
  #     aten.detach: coo_detach,
  #     aten.to_dense: coo_to_dense,
  #     aten._to_copy: coo_to_copy,
  #     aten._nnz: coo_nnz,
  #     aten.sparse_dim: coo_sparse_dim,
  #     aten.to: coo_to,
  # }
  _v: torch.Tensor
  _i: torch.Tensor
  # make these constant properties
  _layout: torch.layout = torch.sparse_coo
  is_sparse: bool = True
  is_coalesced: bool = False

  __slots__ = ('_v', '_i', '_layout_key')

  def __new__(cls,
              indices: torch.Tensor,
              values: torch.Tensor,
              size: Iterable[int],
              *,
              dtype: torch.dtype | None = None,
              device: torch.device | None = None,
              requires_grad: bool = False):
    device = device if device is not None else values.device
    dtype = dtype if dtype is not None else values.dtype
    # We must use the autograd function to create in an autograd context, see [_coo_ops.make_sparse, or SparesCOOTensor.create]
    assert not requires_grad or not values.requires_grad or not torch.is_grad_enabled(
    )
    assert (device == values.device and device == indices.device and
            dtype == values.dtype)

    # Passing layout to _make_wrapper_subclass will trigger the Sparse modifier
    # being added to the dispatch key set. We want to avoid this so that on
    # dispatch we don't hit an overlaod resolution error if things try to hit
    # the sparse ctors before python dispatch.
    return torch.Tensor._make_wrapper_subclass(
        cls, size=size, strides=None, dtype=dtype, device=device)

  def __init__(self,
               indices: torch.Tensor,
               values: torch.Tensor,
               size: Iterable[int],
               *,
               dtype: torch.dtype | None = None,
               device: torch.device | None = None,
               requires_grad: bool = False,
               is_coalesced: bool = False):
    self._v = values
    self._i = indices
    # note is_coalesced is ignored if false and coalesced state is trival (0/1 nnz)
    self._is_coalesced = is_coalesced or self._nnz() < 2

  # Without this the default __torch_function__ impl will attempt to wrap any
  # returned tensors as our subtype which is wrong.  We handle creating sparse
  # returns where needed, and bare tensors returned should remain as such.
  __torch_function__ = torch._C._disabled_torch_function_impl

  # @classmethod
  # def __torch_function__(cls,
  #                        func: OpOverload,
  #                        types: Tuple,
  #                        args: Tuple = (),
  #                        kwargs: Dict[str, Any] | None = None) -> torch.Tensor:
  #   kwargs = kwargs or {}
  #   from ._coo_ops import _COO_FUNCTION_TABLE
  #   if func in _COO_FUNCTION_TABLE:
  #     return _COO_FUNCTION_TABLE[func](args=args, kwargs=kwargs)
  #   else:
  #     super().__torch_function__(func, types, args, kwargs)

  @classmethod
  def __torch_dispatch__(cls,
                         func: OpOverload,
                         types: Tuple,
                         args: Tuple = (),
                         kwargs: Dict[str, Any] | None = None) -> torch.Tensor:
    func = func.overloadpacket
    from ._coo_ops import _COO_DISPATCH_TABLE
    print(func.__name__)
    if func in _COO_DISPATCH_TABLE:
      return _COO_DISPATCH_TABLE[func](args, kwargs)
    msg = (f"Sparse support via {cls.__name__} is limited to a very narrow "
           f"scope. The {func.__name__} operator is not currently supported. It"
           " is likely you are using this outside of its intended purpose "
           "if you see this message. Supported operators are: "
           f"{list(k.__name__ for k in _COO_DISPATCH_TABLE.keys())}")
    warnings.warn(msg)
    return NotImplemented

  @classmethod
  def create(cls, *args, **kwargs):
    from ._coo_ops import make_sparse
    return make_sparse(*args, **kwargs)

  @property
  def layout(self):
    return self._layout
