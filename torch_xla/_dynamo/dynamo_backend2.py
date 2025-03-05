from typing import Any
import torch

def dynamo_backend(model: torch.fx.GraphModule, sample_args: Any):
  """Takes fx graph as input, returns a callable."""

  try:
    import torchax
    from torchax.export import JaxInterpreter
  except ImportError:
    print('To use this dynamo backend, please install torchax')
    raise

  def run_jax(*args):
    res = JaxInterpreter(model).run(
      *args, enable_io_processing=False)
    return res

  computation = xb.jax_func_to_xla_computation(run_jax, args, {}, 'dynamo_jax')
  
  def equivalent(*args, **kwargs):
    flattened = pytree.tree_flatten((args, kwargs))
    return computation(flattened)

  return equivalent


