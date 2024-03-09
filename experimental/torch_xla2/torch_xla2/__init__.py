import jax
import torch
from torch._functorch import make_functional
from torch.utils import _pytree as pytree
from torch_xla2 import tensor
from torch_xla2 import export, _ops, ops_registry, tensor, tf_integration



def extract_jax(mod: torch.nn.Module):
  """Returns a pytree of jax.ndarray and a jax callable."""
  func, weights, buffer = make_functional.make_functional_with_buffers(mod)
  states = (weights, buffer)
  states = pytree.tree_map_only(torch.Tensor, tensor.t2j, states)

  #@jax.jit
  def jax_func(states, inputs):
    (states, inputs) = tensor.wrap((states, inputs))
    weights, buffer = states
    with tensor.XLADispatchMode():
      res = func(weights, buffer, *inputs)
    return tensor.unwrap(res)

  return states, jax_func
