import torch
from .version import __version__
import _XLAC

_XLAC._initialize_aten_bindings()
