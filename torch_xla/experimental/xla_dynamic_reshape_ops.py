from typing import Any, List

import torch
import torch_xla
from torch.library import Library, impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "dynamic_expand(Tensor input, SymInt[] size, Tensor src_tensor, int src_dim, int target_dim) -> Tensor"
)


@impl(XLA_LIB, "dynamic_expand", "XLA")
def dynamic_expand_xla(
    input: torch.Tensor,
    size: List[Any],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
):
  """torch.expand with unbounded dynamic input shape.
  
     At most one dim of output shape can be unbounded dynamic.
     A static dim of input tensor can be expanded to an unbounded dynamic size.
     The unbounded dim of input cannot be expanded to a different unbounded size. 
    
    Args:
        input: torch.Tensor - input tensor to be expanded.
        size: List[Any] - Expanded size.
        src_tensor: torch.Tensor - Tensor with the unbounded dimension size to which the input will be expand.
        src_dim: int - The src_tensor dimension served as the unbounded size in the expanded shape.
        target_dim: int - The dimension of the output tensor is unbounded dynamic.
    """
  return torch_xla._XLAC._xla_dynamic_expand(input, size, src_tensor, src_dim,
                                             target_dim)


@impl(XLA_LIB, "dynamic_expand", "CompositeExplicitAutograd")
def dynamic_expand(
    input: torch.Tensor,
    size: List[Any],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
):
  size[target_dim] = src_tensor.shape[src_dim]
  return input.expand(*size)


@impl(XLA_LIB, "dynamic_expand", "Meta")
def dynamic_expand_meta(
    input: torch.Tensor,
    size: List[Any],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
):
  final_size = list(input.shape)
  final_size[target_dim] = src_tensor.shape[src_dim]
  return torch.empty(*final_size, device="meta")


XLA_LIB.define(
    "dynamic_view(Tensor input, SymInt[] size, Tensor src_tensor, int src_dim, int target_dim, float mul_scaler) -> Tensor"
)


@impl(XLA_LIB, "dynamic_view", "XLA")
def dynamic_view_xla(
    input: torch.Tensor,
    size: List[Any],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
    mul_scaler: int,
):
  """torch.view with unbounded dynamic input shape.
  
     At most one dim of output shape can be unbounded dynamic.
     The unbounded dimension size can be the same,
     or scaled by an integer factor.

    Args:
        input: torch.Tensor - input tensor.
        size: List[Any] - Output size.
        src_tensor: torch.Tensor - Tensor serving as the source of the unbounded dim size.
        src_dim: int - The src_tensor dimension served as the unbounded size in the output shape.
        target_dim: int - The dimension of the output tensor is unbounded dynamic.
        mul_scaler: int - scale factor of the unbounded dynamic dimension size of src_tensor.shape[src_dim]
    """
  return torch_xla._XLAC._xla_dynamic_view(input, size, src_tensor, src_dim,
                                           target_dim, mul_scaler)


@impl(XLA_LIB, "dynamic_view", "CompositeExplicitAutograd")
def dynamic_view(
    input: torch.Tensor,
    size: List[Any],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
    mul_scaler: int,
):
  size[target_dim] = src_tensor.shape[src_dim] * int(mul_scaler)
  return input.view(size)


@impl(XLA_LIB, "dynamic_view", "Meta")
def dynamic_view_meta(
    input: torch.Tensor,
    size: List[Any],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
    mul_scaler: int,
):
  new_dims = list(size)
  new_dims[target_dim] = src_tensor.shape[src_dim] * int(mul_scaler)
  return input.view(new_dims)
