import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
from typing import Any
import warnings


class autocast(torch.amp.autocast_mode.autocast):
  r"""
  `torch.autocast` for XLA backend devices. See :class:`torch.autocast`.
  ``torch_xla.amp.autocast(device, **kwargs)`` is equivalent to
  ``torch.autocast("xla", **kwargs)`` for XLA:TPU for dtype torch.bfloat16.
  """

  def __init__(self,
               device,
               enabled: bool = True,
               dtype: torch.dtype = None,
               cache_enabled: bool = True):
    # `torch_xla.amp.autocast` is intended for XLA backend, with AutocastXLA dispatch key.
    assert 'xla' in str(device), "torch_xla.autocast is available for XLA:TPU"

    self._enabled = enabled
    self._xla_device = xm.xla_device_hw(device)
    if self._xla_device == 'TPU' or self._xla_device == 'NEURON':
      if dtype is None:
        dtype = torch.bfloat16
      if dtype != torch.bfloat16:
        error_message = "In XLA:TPU autocast, but the target dtype is not supported. Disabling autocast.\n"
        error_message += (
            "TPU Autocast only supports dtype of torch.bfloat16 currently.")
        warnings.warn(error_message)
        enabled = False
      self._dtype = dtype
      super().__init__(
          "xla",
          enabled=enabled,
          dtype=self._dtype,
          cache_enabled=cache_enabled)
    else:
      print('Warning: AMP only supported for XLA:TPU. Ignoring autocast.')
