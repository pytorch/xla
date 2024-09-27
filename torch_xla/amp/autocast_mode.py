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
  ``torch.autocast("xla", **kwargs)`` for XLA:GPU and XLA:TPU for dtype torch.bfloat16,
  ``torch.autocast("cuda", **kwargs)`` for XLA:GPU and other dtypes.
  """

  def __init__(self,
               device,
               enabled: bool = True,
               dtype: torch.dtype = None,
               cache_enabled: bool = True):
    # `torch_xla.amp.autocast` is intended for XLA backend, with AutocastXLA dispatch key.
    assert 'xla' in device.__str__(
    ), "torch_xla.autocast is available for XLA:TPU, XLA:GPU"

    self._enabled = enabled
    self._xla_device = xm.xla_device_hw(device)
    if self._xla_device == 'CUDA':
      backend = 'cuda'
      self._xla_bfloat16 = False  # True if xla backend with bfloat16 dtype.
      if dtype is None:
        dtype = torch.float16
      elif dtype == torch.bfloat16 and not torch.cuda.is_available():
        if xr.is_bf16_supported():
          # XLA:GPU with bfloat16 should run on `xla` backend
          # unless torch.autocast is compiled with cuda.
          backend = 'xla'
          self._xla_bfloat16 = True
        else:
          # This has been the default behavior for unsupported bfloat16 dtype
          dtype = torch.float16
          error_message = "In XLA:GPU autocast, but bfloat16 is not supported on this HW.\n"
          error_message += ("Using the default cuda autocast dtype float16.")
      self._dtype = dtype
      super().__init__(
          backend,
          enabled=enabled,
          dtype=self._dtype,
          cache_enabled=cache_enabled)
    elif self._xla_device == 'TPU' or self._xla_device == 'NEURON':
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
      print(
          'Warning: AMP only supported for XLA:TPU and XLA:GPU. Ignoring autocast.'
      )

  def __enter__(self):
    # This ensures that xla autocast is enabled even for XLA:GPU, which calls
    # `torch.amp.autocast_mode.autocast` with `cuda` backend.
    if self._xla_device == 'CUDA':
      self.prev = torch.is_autocast_xla_enabled()  # type: ignore[attr-defined]
      self.prev_dtype = torch.get_autocast_xla_dtype(
      )  # type: ignore[attr-defined]
      if self._xla_bfloat16:
        # autocast_xla flags will be set by `torch.autocast` and we need to
        # set autocast flags as we call into `torch.autocast` apis.
        torch.set_autocast_enabled(self._enabled)
        torch.set_autocast_gpu_dtype(self._dtype)
      else:
        torch.set_autocast_xla_enabled(self._enabled)
        torch.set_autocast_xla_dtype(self._dtype)
    return super().__enter__()

  def __exit__(self, exc_type: Any, exc_val: Any,
               exc_tb: Any):  # type: ignore[override]
    if self._xla_device == 'CUDA':
      if self._xla_bfloat16:
        # autocast_xla flags will be set by `torch.autocast` and we need to
        # set autocast flags as we call into `torch.autocast` apis.
        torch.set_autocast_enabled(self.prev)
        torch.set_autocast_gpu_dtype(self.prev_dtype)
      else:
        torch.set_autocast_xla_enabled(self.prev)
        torch.set_autocast_xla_dtype(self.prev_dtype)
    return super().__exit__(exc_type, exc_val, exc_tb)

  def __call__(self, func):
    return super().__call__(func)
