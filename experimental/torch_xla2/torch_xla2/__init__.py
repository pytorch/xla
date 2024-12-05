import contextlib
from typing import List, Dict, Any, Optional
import dataclasses
import jax
import os
import torch
from torch.utils import _pytree as pytree
from torch_xla2 import tensor
from torch_xla2 import distributed  # noqa: F401

__version__ = "0.0.1"
VERSION = __version__

__all__ = [
  'default_env',
  'extract_jax',
  'enable_globally',
]

from jax._src import xla_bridge
os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')

# torch_xla2:oss-begin
old_pjrt_options = jax.config.jax_pjrt_client_create_options
try:
  jax.config.update(
    'jax_pjrt_client_create_options',
    f'ml_framework_name:PyTorch/XLA2;ml_framework_version:{"v0.0.1"}'
  )
  xla_bridge._clear_backends()
  if os.environ.get("DISABLE_XLA2_PJRT_TEST") != "true":
    jax.devices()  # open PJRT  to see if it opens
except RuntimeError:
  jax.config.update(
    'jax_pjrt_client_create_options', old_pjrt_options
  )
  xla_bridge._clear_backends()
  jax.devices()  # open PJRT  to see if it opens
# torch_xla2:oss-end

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
  states = mod.state_dict()

  states = pytree.tree_map_only(torch.Tensor, tensor.t2j, states)

  #@jax.jit
  def jax_func(states, inputs):
    (states, inputs) = env.j2t_iso((states, inputs))
    with env:
      res = torch.func.functional_call(mod, states, inputs, tie_weights=False)
    return env.t2j_iso(res)

  return states, jax_func

def enable_globally():
  global env 
  env = default_env().__enter__()
  return env

def disable_globally():
  global env 
  default_env().__exit__(None, None, None)

@contextlib.contextmanager
def disable_temporarily():
  prev = default_env().enabled
  if prev:
    disable_globally()
  yield()
  if prev:
    enable_globally()


torch.utils.rename_privateuse1_backend('jax')
unsupported_dtype = [torch.quint8]
torch.utils.generate_methods_for_privateuse1_backend(
  for_tensor=True, for_module=True, for_storage=True, 
  unsupported_dtype=unsupported_dtype)

import jax
import torch_xla2.device_module
torch._register_device_module('jax', torch_xla2.device_module)




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
  methods_to_compile: List[str] = dataclasses.field(default_factory=lambda: ['forward'])  
  jax_jit_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
  mode: str = 'jax' # or dynamo or export


def compile(fn, options: Optional[CompileOptions] = None):
  options = options or CompileOptions()
  if options.mode == 'jax':
    from torch_xla2 import interop
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
