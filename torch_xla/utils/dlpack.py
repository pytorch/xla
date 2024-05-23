from typing import Any
import enum
import torch_xla


def to_dlpack(xla_tensor: Any):
  return torch_xla._XLAC._to_dlpack(xla_tensor)

class DLDeviceType(enum.IntEnum):
    # Enums as in DLPack specification (aten/src/ATen/dlpack.h)
    kDLCPU = 1,
    kDLGPU = 2,
    kDLCPUPinned = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLExtDev = 12,
    kDLOneAPI = 14,

def from_dlpack(ext_tensor: Any):
  print('xw32 from_dlpack line23 is called')
  if hasattr(ext_tensor, '__dlpack_device__') and hasattr(ext_tensor, '__dlpack__'):
    device_type, device_id = ext_tensor.__dlpack_device__()
    print('xw32 cuda tensor have the __dlpack__ attr, device_type=', device_type)
    if device_type == DLDeviceType.kDLGPU:
      stream = torch_xla._XLAC._get_stream_for_cuda_device(device_id)
      print('xw32 got cuda stream:', stream)
      dlpack = ext_tensor.__dlpack__(stream=stream)
    else:
      dlpack = ext_tensor.__dlpack__()
  else:
    print('xw32 cuda tensor doesnt have the __dlpack__ attr')
    dlpack = ext_tensor

  return torch_xla._XLAC._from_dlpack(dlpack)
