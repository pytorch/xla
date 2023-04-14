import torch
import torch_xla.core.xla_model as xm
from typing import Any


class autocast(torch.amp.autocast_mode.autocast):
  r"""
    See :class:`torch.autocast`.
    ``torch_xla.amp.autocast(device, args...)`` is equivalent to ``torch.autocast("xla", args...)`` for TPUs
    ``torch.autocast("cuda", args...)`` for GPUs.
    """

  def __init__(self,
               device,
               enabled: bool = True,
               dtype: torch.dtype = torch.bfloat16,
               cache_enabled: bool = True):
    if xm.xla_device_hw(device) == 'GPU':
      super().__init__(
          "cuda",
          enabled=enabled,
          dtype=torch.float16,
          cache_enabled=cache_enabled)
    else:
      super().__init__(
          "xla", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

  def __enter__(self):
    return super().__enter__()

  def __exit__(self, exc_type: Any, exc_val: Any,
               exc_tb: Any):  # type: ignore[override]
    return super().__exit__(exc_type, exc_val, exc_tb)

  def __call__(self, func):
    return super().__call__(func)
