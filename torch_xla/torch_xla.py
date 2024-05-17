import contextlib
from typing import List

import torch
import torch_xla
import torch_xla.core.xla_model as xm


def device(index: int = None) -> torch.device:
  """Returns a given instance of an XLA device.

  If SPMD enables, returns a virtual device that wraps all devices available
  to this process.

  Args:
    index: index of the XLA device to be returned. Corresponds to index in
      `torch_xla.devices()`.

  Returns:
    An XLA `torch.device`.
  """

  return xm.xla_device(index)


def devices() -> List[torch.device]:
  """Returns all devices available in the current process.

  Returns:
    A list of XLA `torch.devices`.
  """

  return [torch.device(d) for d in xm.get_xla_supported_devices()]


def real_devices() -> List[str]:
  """Returns local XLA device types and indices.

  Returns:
    A list strings representing the XLA devices available in the current
    process, e.g. `['TPU:0', 'TPU:1', ...]`.
  """

  return torch_xla._XLAC._xla_real_devices()


def device_count() -> int:
  """Returns number of addressable devices in the current process."""
  return len(real_devices())


def sync():
  """Launches all pending graph operations."""
  xm.mark_step()


@contextlib.contextmanager
def step():
  """Wraps code that should be dispatched to the runtime.

  Experimental: `xla.step` is still a work in progress. Some code that currently
  works with `xla.step` but does not follow best practices will become errors in
  future releases. See https://github.com/pytorch/xla/issues/6751 for context.
  """
  # Clear pending operations
  xm.mark_step()

  try:
    yield
  finally:
    xm.mark_step()
