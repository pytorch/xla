from typing import Any
import torch_xla

def to_dlpack(xla_tensor: Any):
  dlt = torch_xla._XLAC._to_dlpack(xla_tensor)
  print('xw32 torch_xla._XLAC._to_dlpack has returned.')
  return dlt

def from_dlpack(ext_tensor: Any):
  return torch_xla._XLAC._from_dlpack(ext_tensor)
