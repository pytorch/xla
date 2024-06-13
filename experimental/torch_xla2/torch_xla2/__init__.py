import jax
import os
import torch
from torch.utils import _pytree as pytree
from torch_xla2 import tensor


__all__ = [
  'default_env',
  'extract_jax',
]


jax.config.update('jax_enable_x64', True)
jax.config.update(
  'jax_pjrt_client_create_options',
  f'ml_framework_name:PyTorch/XLA2;ml_framework_version:{"v0.0.1"}'
)

env = None
def default_env():
  global env

  os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')

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
      res = torch.func.functional_call(mod, states, inputs)
    return env.t2j_iso(res)

  return states, jax_func
