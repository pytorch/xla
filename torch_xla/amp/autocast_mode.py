import torch
import torch_xla.core.xla_model as xm
from typing import Any
import warnings


class autocast(torch.amp.autocast_mode.autocast):
  r"""
  `torch.autocast` for XLA backend devices. See :class:`torch.autocast`.
  ``torch_xla.amp.autocast(device, **kwargs)`` is equivalent to
  ``torch.autocast("xla", **kwargs)`` for XLA:TPU and XLA:GPU backends.
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
    if self._xla_device is 'GPU':
      # TODO(yeounoh) support torch.bfloat16 in XLA:GPU for eligible HW
      if dtype is None:
        dtype = torch.float16
      if dtype == torch.bfloat16:
        error_message = "In XLA:GPU autocast, bfloat16 is currently not supported.\n"
        error_message += ("It is treated as float16, which is the default XLA:GPU autocast dtype.")
        warnings.warn(error_message)
        # This has been the default behavior for XLA:GPU, since r2.0
        dtype = torch.float16
      self._dtype = dtype
      super().__init__(
          "cuda", enabled=enabled, dtype=self._dtype, cache_enabled=cache_enabled)
    elif self._xla_device is 'TPU':
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
          "xla", enabled=enabled, dtype=self._dtype, cache_enabled=cache_enabled)
    else:
      print(
          'Warning: AMP only supported for XLA:TPU and XLA:GPU. Ignoring autocast.'
      )

  def __enter__(self):
    # This ensures that xla autocast is enabled even for XLA:GPU, which calls
    # `torch.amp.autocast_mode.autocast` with `cuda` backend.
    if self._xla_device is 'GPU':
      self.prev = torch.is_autocast_xla_enabled()  # type: ignore[attr-defined]
      self.prev_dtype = torch.get_autocast_xla_dtype()  # type: ignore[attr-defined]
      torch.set_autocast_xla_enabled(self._enabled)
      torch.set_autocast_xla_dtype(self._dtype)
    return super().__enter__()

  def __exit__(self, exc_type: Any, exc_val: Any,
               exc_tb: Any):  # type: ignore[override]
    if self._xla_device is 'GPU':
      torch.set_autocast_xla_enabled(self.prev)
      torch.set_autocast_xla_dtype(self.prev_dtype)
    return super().__exit__(exc_type, exc_val, exc_tb)

  def __call__(self, func):
    return super().__call__(func)
