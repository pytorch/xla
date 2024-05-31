from typing import Any
import enum
from torch.utils.dlpack import DLDeviceType
import torch_xla


def to_dlpack(xla_tensor: Any):
  return torch_xla._XLAC._to_dlpack(xla_tensor)


def from_dlpack(ext_tensor: Any):
  if hasattr(ext_tensor, '__dlpack_device__') and hasattr(
      ext_tensor, '__dlpack__'):
    device_type, device_id = ext_tensor.__dlpack_device__()
    if device_type == DLDeviceType.kDLGPU:
      stream = torch_xla._XLAC._get_stream_for_cuda_device(device_id)
      dlpack = ext_tensor.__dlpack__(stream=stream)
    else:
      dlpack = ext_tensor.__dlpack__()
  else:
    dlpack = ext_tensor

  return torch_xla._XLAC._from_dlpack(dlpack)
