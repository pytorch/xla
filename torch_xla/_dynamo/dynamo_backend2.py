import functools
from typing import Any
import torch
from torch.utils import _pytree as pytree
from torch_xla.core import xla_builder as xb
import torch_xla

from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func


def _dynamo_backend(model: torch.fx.GraphModule, sample_args: Any):
  """A dynamo backend that compiles a FX graph to HLO using JAX and torchax.

  It takes FX graph as input and returns a compiled PyTorch function. The FX graph
  is traced into a JAX function using torchax, and the JAX function is lowered to HLO.

  Args:
    model: the graph to be compiled
    sample_args: a tuple or list of sample inputs. I.e. model(*sample_args) produces
      the model output

  Returns:
    Another callable f such that f(*sample_inputs) computes the same thing as model.
  """

  try:
    import torchax.interop
    from torchax.export import JaxInterpreter
    import jax
  except ImportError:
    print('To use this dynamo backend, please install torchax')
    raise

  jax.config.update("jax_enable_x64", True)
  env = torchax.default_env()
  xla_device = torch_xla.device()

  def run_jax(*args, initial_rng_key):
    args_t = torchax.interop.torch_view(args)
    env.manual_seed(initial_rng_key)
    with env:
      res = model(*args_t)
    return torchax.interop.jax_view(res)

  initial_rng_key = torch.tensor(0, device=xla_device, dtype=torch.uint32)

  computation = xb.JaxCallable(run_jax)

  def equivalent(*args, **kwargs):
    kwargs['initial_rng_key'] = torch.randint(
        0, 2**32, (), dtype=torch.uint32, device=xla_device)
    res = computation(*args, **kwargs)
    return res

  return make_boxed_func(equivalent)


def dynamo_backend(fx, args):
  from functorch.compile import aot_function
  return aot_function(fx, fw_compiler=_dynamo_backend)
