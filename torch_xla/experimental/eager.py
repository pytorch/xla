import functools
from contextlib import contextmanager

import torch_xla
import logging


def eager_mode(enable: bool):
  """Configure torch_xla's default executation mode.

  Under eager mode only functions that was `torch_xla.compile`d will be
  traced and compiled. Other torch ops will be executed eagerly.
  """
  torch_xla._XLAC._set_use_eager_mode(enable)


def is_eager_mode() -> bool:
  """Return True if torch_xla is currently under eager mode
  """
  return torch_xla._XLAC._get_use_eager_mode()


@contextmanager
def eager_mode_context(enable: bool):
  """Context manager to enable/disable the eager mode.
  """
  saved_eager_mode = is_eager_mode()
  eager_mode(enable)
  try:
    yield saved_eager_mode
  finally:
    eager_mode(saved_eager_mode)


def compile(func):
  # can's use deprecated wrapper at import time due to circular dependency
  logging.warning(
      'torch_xla.experimental.compile is deprecated. Use torch_xla.compile instead.'
  )
  return torch_xla.compile(func)
