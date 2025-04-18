from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Tuple
import torch

from torch._ops import OpOverload

from ._coo_ops import (
    coo_clone,
    coo_sparse_mask,
    coo_resize_as_,
    coo_coalesce,
    coo_add_,
    coo_indices,
    coo_values,
    coo_indices_unsafe,
    coo_values_unsafe,
    coo_detach,
    fallback_dispatch,
)

__all__ = ["SparseCOOTensor"]

aten = torch.ops.aten


class SparseCOOTensor(torch.Tensor):

  DISPATCH_TABLE: ClassVar[Dict[OpOverload, Callable]] = {
      aten.values: coo_values,
      aten.indices: coo_indices,
      aten._indices: coo_indices_unsafe,
      aten._values: coo_values_unsafe,
      aten.coalesce: coo_coalesce,
      aten.add_: coo_add_,
      aten.sparse_mask: coo_sparse_mask,
      aten.resize_as_: coo_resize_as_,
      aten.clone: coo_clone,
      aten._to_copy: fallback_dispatch(aten._to_copy),
      aten.detach: coo_detach,
  }
  _v: torch.Tensor
  _i: torch.Tensor
  # make these constant properties
  layout: torch.layout
  is_sparse = True

  __slots__ = ('_v', '_i', '_layout_key')

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

    # Passing layout to _make_wrapper_subclass will trigger the Sparse modifier
    # being added to the dispatch key set. We want to avoid this so that on
    # dispatch we don't hit an overlaod resolution error if things try to hit
    # the sparse ctors before python dispatch.
    res = torch.Tensor._make_wrapper_subclass(
        cls,
        size=size,
        strides=None,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad)
    res._v = values
    res._i = indices
    res._layout_key = torch.sparse_coo
    return res

  @classmethod
  def __torch_dispatch__(cls,
                         func: OpOverload,
                         types: Tuple,
                         args: Tuple = (),
                         kwargs: Dict[str, Any] | None = None) -> torch.Tensor:
    impl = cls._match_func_key_for_dispatch(func)
    return impl(args, kwargs)

  #__torch_function__ = torch._C._disabled_torch_function_impl

  def __tensor_flatten__(
      self) -> Tuple[list[str], Tuple[torch.Size, torch.layout, bool]]:
    inner_tensors = ['_v', '_i']
    tensor_meta = (self.shape, self.requires_grad, self._layout_key)
    return inner_tensors, tensor_meta

  @classmethod
  def __tensor_unflatten__(
      cls,
      inner_tensors,
      tensor_meta: Tuple[torch.Size, torch.layout, bool],
  ) -> Tuple[list[str], Tuple[torch.Size, torch.layout, bool]]:
    size, layout, requires_grad = tensor_meta
    return cls(
        inner_tensors.get("_i"),
        inner_tensors.get('_v'),
        size=size,
        layout=layout,
        requires_grad=requires_grad)

  @classmethod
  def _match_func_key_for_dispatch(cls, func):
    impl = cls.DISPATCH_TABLE.get(func, None)
    if impl is None:
      impl = cls.DISPATCH_TABLE.get(func._overloadpacket,
                                    cls._make_not_implemented(func))
    return impl

  @classmethod
  def _make_not_implemented(cls, func):

    def impl(args=(), kwargs=None):
      raise NotImplementedError(
          f"Sparse support via {cls.__name__} is limited to a very narrow "
          f"scope. The {func.__name__} operator is not currently supported. It"
          " is likely you are using this outside of its intended purpose "
          "if you see this message. Supported operators are: ",
          f"{list(k.__name__ for k in cls.DISPATCH_TABLE.keys())}")

    return impl
