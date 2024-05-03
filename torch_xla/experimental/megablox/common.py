"""Common utilities for Pallas kernels."""

from typing import Union
import torch
import tpu_features


def assert_is_supported_dtype(dtype: torch.dtype) -> None:
  if dtype != torch.bfloat16 and dtype != torch.float32:
    raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.dtype:
  """A type to which both input should be adapted to before dot product."""
  # bf16xbf16 matmul is only supported since TPU v4 generation. In
  # case of mixed input precision, we need to convert bf16 argument to fp32
  # beforehand.
  if (tpu_features.supports_bfloat16_matmul() and
      lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16):
    return torch.bfloat16
  else:
    return torch.float32
