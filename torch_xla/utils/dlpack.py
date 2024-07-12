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
    device_type, device_id = ext_tensor.__dlpack_device__()
    if device_type == DLDeviceType.kDLGPU:
      stream = torch_xla._XLAC._get_stream_for_cuda_device(device_id)
      dlpack = ext_tensor.__dlpack__(stream=stream)
    else:
      dlpack = ext_tensor.__dlpack__()
  else:
    dlpack = ext_tensor

  return torch_xla._XLAC._from_dlpack(dlpack)


def from_xla_cuda_to_cuda(tensor):
  assert torch.cuda.is_available()
  assert tensor.device.type == "xla", "The tensor is not an XLA tensor"
  is_xla_cuda = True if xu.getenv_as("PJRT_DEVICE", str,
                                     "").lower() == "cuda" else False
  assert is_xla_cuda, "The XLA tensor is not on CUDA"
  # consumer is torch, producer is torch_xla

  # Similar logic as torch.utils.dlpack.from_dlpack
  # https://github.com/pytorch/pytorch/blob/b0ef363972203b163cddc95e4c6054b8221c2300/torch/utils/dlpack.py#L114-L115
  # The array API specify that the default legacy stream must be passed
  # with a value of 1 for CUDA
  device_id = tensor.device.index
  stream = torch_xla._XLAC._get_stream_for_cuda_device(device_id)
  stream = 1 if stream == 0 else stream
  assert stream is None or type(stream) is int
  external_stream = torch.cuda.ExternalStream(stream)
  current_stream = torch.cuda.current_stream()
  if external_stream != current_stream:
    event = torch.cuda.Event()
    event.record(current_stream)
    external_stream.wait_event(event)
  dlpack = to_dlpack(tensor)
  cuda_tensor = torch.utils.dlpack.from_dlpack(dlpack)
  return cuda_tensor
