from typing import Any
import torch
from torch.utils import _pytree as pytree
from torch_xla.core import xla_builder as xb

def dynamo_backend(model: torch.fx.GraphModule, sample_args: Any):
  """Takes fx graph as input, returns a callable."""

  try:
    import torchax.interop
    from torchax.export import JaxInterpreter
  except ImportError:
    print('To use this dynamo backend, please install torchax')
    raise

  def run_jax(*args):
    args_t = torchax.interop.torch_view(args)
    res = model(*args_t)
    return torchax.interop.jax_view(res)

  computation = xb.jax_func_to_xla_computation(
    run_jax, sample_args, {}, 'dynamo_jax')
  
  def equivalent(*args, **kwargs):
    flattened, _ = pytree.tree_flatten((args, kwargs))
    res = computation(flattened)
    if not isinstance(res, (list, tuple)):
      return (res, )
    return res

  return equivalent


