from typing import Any
import torch_xla

def to_dlpack(xla_tensor: Any):
  return torch_xla._XLAC._to_dlpack(xla_tensor)
  # dlt = torch_xla._XLAC._to_dlpack(xla_tensor)
  # print('xw32 torch_xla._XLAC._to_dlpack has returned. dlt has __dlpack_+=', hasattr(dlt, "__dlpack__"), ', dlt has __dlpack_device__=', hasattr(dlt, "__dlpack_device__"))
  # return dlt

def from_dlpack(ext_tensor: Any):
  return torch_xla._XLAC._from_dlpack(ext_tensor)
