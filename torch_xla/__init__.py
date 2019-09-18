import torch
import _XLAC

try:
  from .version import __version__
except ImportError:
  # Only wheel installation has torch_xla.version module
  __version__ = ''

_XLAC._initialize_aten_bindings()
