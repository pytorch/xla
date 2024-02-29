from typing import List

import torch
import torch_xla
from torch.library import Library, impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "dynamic_expand(Tensor x, int[] size, Tensor src_tensor, int src_dim, int target_dim) -> Tensor"
)


@impl(XLA_LIB, "dynamic_expand", "XLA")
def dynamic_expand_xla(
    x: torch.Tensor,
    size: List[int],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
):
  """Attach pattern boundary metadata to a XLA Tensor.
    Args:
        x: torch.Tensor (On XLA device) - the marked tensor.
        name: str - The name of the pattern, it will be the name of the stablehlo composite op.
        pos: int - Input/output Position of the annotated tensor in the pattern.
        id: str - Unique identifier of the pattern instance.
        is_input: bool - If the annotated tensor is the input to the pattern.
        attr: dict - Attribute of the pattern, it will be passed down to the attribute field
                     in the stablehlo composite.
    """
  return torch_xla._XLAC._xla_dynamic_expand(x, size, src_tensor, src_dim,
                                             target_dim)


@impl(XLA_LIB, "dynamic_expand", "CompositeExplicitAutograd")
def dynamic_expand(
    x: torch.Tensor,
    size: List[int],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
):
  size[target_dim] = src_tensor.shape[src_dim]
  return x.expand(*size)


@impl(XLA_LIB, "dynamic_expand", "Meta")
def dynamic_expand_meta(
    x: torch.Tensor,
    size: List[int],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
):
  final_size = list(x.shape)
  final_size[target_dim] = src_tensor.shape[src_dim]
  return torch.empty(*final_size, device="meta")


XLA_LIB.define(
    "dynamic_view(Tensor x, int[] size, Tensor src_tensor, int src_dim, int target_dim, float mul_scaler) -> Tensor"
)


@impl(XLA_LIB, "dynamic_view", "XLA")
def dynamic_view_xla(
    x: torch.Tensor,
    size: List[int],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
    mul_scaler: int,
):
  """Attach pattern boundary metadata to a XLA Tensor.
    Args:
        x: torch.Tensor (On XLA device) - the marked tensor.
        name: str - The name of the pattern, it will be the name of the stablehlo composite op.
        pos: int - Input/output Position of the annotated tensor in the pattern.
        id: str - Unique identifier of the pattern instance.
        is_input: bool - If the annotated tensor is the input to the pattern.
        attr: dict - Attribute of the pattern, it will be passed down to the attribute field
                     in the stablehlo composite.
    """
  return torch_xla._XLAC._xla_dynamic_view(x, size, src_tensor, src_dim,
                                           target_dim, mul_scaler)


@impl(XLA_LIB, "dynamic_view", "CompositeExplicitAutograd")
def dynamic_view(
    x: torch.Tensor,
    size: List[int],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
    mul_scaler: int,
):
  size[target_dim] = src_tensor.shape[src_dim] * int(mul_scaler)
  return x.view(size)


@impl(XLA_LIB, "dynamic_view", "Meta")
def dynamic_view_meta(
    x: torch.Tensor,
    size: List[int],
    src_tensor: torch.Tensor,
    src_dim: int,
    target_dim: int,
    mul_scaler: int,
):
  new_dims = list(size)
  new_dims[target_dim] = src_tensor.shape[src_dim] * int(mul_scaler)
  return x.view(new_dims)
  # # print(x.shape)
  # size[target_dim] = src_tensor.shape[src_dim] * int(mul_scaler)
  # # print(size)
  # ret = x.view(size)
  # # print(ret.shape)
  # return ret
