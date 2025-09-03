from typing import Any
import enum
from torch.utils.dlpack import DLDeviceType
import torch
import torch_xla
import torch_xla.utils.utils as xu


def to_dlpack(xla_tensor: Any):
  return torch_xla._XLAC._to_dlpack(xla_tensor)


def from_dlpack(ext_tensor: Any):
  if hasattr(ext_tensor, '__dlpack_device__') and hasattr(
      ext_tensor, '__dlpack__'):
    device_type, _ = ext_tensor.__dlpack_device__()
    if device_type != DLDeviceType.kDLCPU:
      raise ValueError(
          "PyTorch/XLA DLPack implementation currently only supports CPU.")
    dlpack = ext_tensor.__dlpack__()
  else:
    dlpack = ext_tensor

  return torch_xla._XLAC._from_dlpack(dlpack)
