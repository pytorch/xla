import os
from contextlib import contextmanager
from typing import Callable, Any
import functools
import logging


# TODO(https://github.com/pytorch/xla/issues/8793): Get rid of this hack.
def jax_import_guard():
  import torch_xla
  # Somehow, we need to grab the TPU before JAX locks it. Otherwise, any pt-xla TPU operations will hang.
  torch_xla._XLAC._init_computation_client()


# TODO(https://github.com/pytorch/xla/issues/8793): Get rid of this hack.
def requires_jax(func: Callable) -> Callable:
  """Decorator that ensures JAX is safely imported before function execution"""

  @functools.wraps(func)
  def wrapper(*args, **kwargs) -> Any:
    try:
      jax_import_guard()
    except ImportError as e:
      raise ImportError(
          "JAX import guard fail due to PJRT client is unavailable.") from e
    with jax_env_context():
      return func(*args, **kwargs)

  return wrapper


# TODO(b/374631442): Get rid of this hack that worksaround MegaScale hanging.
@contextmanager
def jax_env_context():
  previous_skip_megascale_env = None
  try:
    previous_skip_megascale_env = os.environ.get('SKIP_MEGASCALE_PJRT_CLIENT',
                                                 None)
    os.environ['SKIP_MEGASCALE_PJRT_CLIENT'] = 'true'
    yield
  finally:
    if previous_skip_megascale_env:
      os.environ['SKIP_MEGASCALE_PJRT_CLIENT'] = previous_skip_megascale_env
    else:
      os.environ.pop('SKIP_MEGASCALE_PJRT_CLIENT', None)


def maybe_get_torchax():
  try:
    jax_import_guard()
    with jax_env_context():
      import torchax
      import torchax.tensor
      import torchax.interop
      import torchax.ops.mappings
      return torchax
  except (ModuleNotFoundError, ImportError):
    return None


def maybe_get_jax():
  env_val = os.environ.get('TORCH_XLA_ENABLE_JAX', '')
  
  if env_val.lower() in ('1', 'true', 'yes'):
    try:
      jax_import_guard()
      with jax_env_context():
        import jax
        return jax
    except (ModuleNotFoundError, ImportError):
      logging.warning('JAX explicitly enabled but not installed. '
                      'You can install Jax/Pallas via pip install torch_xla[pallas]')
      return None
  
  if env_val == '':
    logging.warning('You are trying to use a feature that requires JAX. '
                    'You can install Jax/Pallas via pip install torch_xla[pallas] and Set TORCH_XLA_ENABLE_JAX=1 to enable JAX features, or TORCH_XLA_ENABLE_JAX=0 to suppress this warning')
  # If explicitly disabled (0, false, no, etc.), return silently
  return None