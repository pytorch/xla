import collections
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

from .xla_flatten_params_wrapper import FlatParameter

_TORCHDISTX_AVAIL = True
try:
  from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
  _TORCHDISTX_AVAIL = False


def _materialize_module(
    module: nn.Module,
    param_init_fn: Optional[Callable[[nn.Module], None]],
    ignored_params: Set[nn.Parameter],
    deferred_init_check_fn: Callable,
) -> bool:
  """
    Materializes the wrapped module ``module`` in place if needed: either
    if the module has parameters that use meta device or are torchdistX
    fake tensors.
    This method uses ``param_init_fn`` to materialize the module if the
    function is not ``None`` and falls back to default behavior otherwise.
    For meta device, this moves the module to ``device_from_device_id`` if
    it is not ``None`` or the current device otherwise and calls
    ``reset_parameters()``, and for torchdistX fake tensors, this calls
    ``deferred_init.materialize_module()``.
    Returns:
        bool: ``True`` if ``module`` was materialized and ``False`` if this was
        a no-op.
    """
  managed_params = list(_get_orig_params(module, ignored_params))
  is_meta_module = any(param.is_meta for param in managed_params)
  is_torchdistX_deferred_init = (
      not is_meta_module and _TORCHDISTX_AVAIL and
      any(fake.is_fake(param) for param in managed_params))
  if (is_meta_module or
      is_torchdistX_deferred_init) and param_init_fn is not None:
    if not callable(param_init_fn):
      raise ValueError(
          f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
      )
    param_init_fn(module)
    return True
  elif is_meta_module:
    # Run default meta device initialization
    module.to_empty(device="cpu")
    try:
      with torch.no_grad():
        module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
      warnings.warn("Unable to call `reset_parameters()` for module on meta "
                    f"device with error {str(e)}. Please ensure your "
                    "module implements a `reset_parameters()` method.")
      raise e
    return True
  elif is_torchdistX_deferred_init:
    # Run default torchdistX initialization
    deferred_init.materialize_module(module, check_fn=deferred_init_check_fn)
    return True
  return False


def _get_orig_params(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Iterator[nn.Parameter]:
  """
    Returns an iterator over the original parameters in ``module``, ignoring
    the parameters in ``ignored_params``, any ``FlatParameter`` s (which may be
    present due to nested FSDP wrapping).
    """
  param_gen = module.parameters()
  try:
    while True:
      param = next(param_gen)
      if param not in ignored_params and not isinstance(param, FlatParameter):
        yield param
  except StopIteration:
    pass
