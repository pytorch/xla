import contextlib
from typing import List, Dict, Any, Optional
import dataclasses
import jax
import os
import torch
from torch.utils import _pytree as pytree
from torchax import tensor
from contextlib import contextmanager

__version__ = "0.0.6"
VERSION = __version__

__all__ = [
    'default_env',
    'extract_jax',
    'enable_globally',
]

from jax._src import xla_bridge

os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')

# torchax:oss-begin
if getattr(jax.config, 'jax_pjrt_client_create_options', None):
  jax.config.update(
      'jax_pjrt_client_create_options',
      f'ml_framework_name:PyTorch/XLA2;ml_framework_version:{"v0.0.1"}')
# torchax:oss-end

env = None


def default_env():
  global env

  if env is None:
    env = tensor.Environment()
  return env


def extract_jax(mod: torch.nn.Module, env=None):
  """Returns a pytree of jax.ndarray and a jax callable."""
  if env is None:
    env = default_env()
  states = dict(mod.named_buffers())
  states.update(mod.named_parameters())

  states = env.t2j_copy(states)

  #@jax.jit
  def jax_func(states, args, kwargs=None):
    (states, args, kwargs) = env.j2t_iso((states, args, kwargs))
    with env:
      res = torch.func.functional_call(
          mod, states, args, kwargs, tie_weights=False)
    return env.t2j_iso(res)

  return states, jax_func


def enable_globally():
  env = default_env().enable_torch_modes()
  return env


def disable_globally():
  global env
  default_env().disable_torch_modes()


@contextlib.contextmanager
def disable_temporarily():
  prev = default_env().enabled
  if prev:
    disable_globally()
  yield ()
  if prev:
    enable_globally()


torch.utils.rename_privateuse1_backend('jax')
unsupported_dtype = [torch.quint8]

import jax
import torchax.device_module

torch._register_device_module('jax', torchax.device_module)


def enable_accuracy_mode():
  jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_default_matmul_precision', 'highest')
  default_env().config.internal_respect_torch_return_dtypes = True


def enable_performance_mode():
  jax.config.update('jax_enable_x64', False)
  jax.config.update('jax_default_matmul_precision', 'default')
  default_env().config.internal_respect_torch_return_dtypes = False


@dataclasses.dataclass
class CompileOptions:
  # only valid if compiling nn.Module
  methods_to_compile: List[str] = dataclasses.field(
      default_factory=lambda: ['forward'])
  jax_jit_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
  mode: str = 'jax'  # or dynamo or export


def compile(fn, options: Optional[CompileOptions] = None):
  options = options or CompileOptions()
  if options.mode == 'jax':
    from torchax import interop
    if isinstance(fn, torch.nn.Module):
      module = interop.JittableModule(fn, extra_jit_args=options.jax_jit_kwargs)
      for n in options.methods_to_compile:
        module.make_jitted(n)
      return module
    else:
      return interop.jax_jit(fn)
  elif options.mode == 'dynamo':
    raise RuntimeError('dynamo mode is not supported yet')
  elif options.mode == 'export':
    raise RuntimeError('export mode is not supported yet')
