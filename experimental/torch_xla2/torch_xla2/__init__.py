import jax
import torch
from torch._functorch import make_functional
from torch.utils import _pytree as pytree
from torch_xla2 import export, tensor, tf_integration

jax.config.update('jax_enable_x64', True)

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
  func, weights, buffer = make_functional.make_functional_with_buffers(mod)
  states = mod.state_dict()

  states = pytree.tree_map_only(torch.Tensor, tensor.t2j, states)

  #@jax.jit
  def jax_func(states, inputs):
    (states, inputs) = env.j2t_iso((states, inputs))
    with env:
      res = torch.func.functional_call(mod, states, inputs)
    return env.t2j_iso(res)

  return states, jax_func
