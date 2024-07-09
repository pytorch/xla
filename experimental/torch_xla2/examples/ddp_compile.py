import functools
import os
from typing import List
import jax
import torch

import torch.distributed as dist
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla2
import torch_xla2.interop
import torch_xla2.export
import numpy as np
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map


class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = nn.Linear(10, 10)
    self.relu = nn.ReLU()
    self.net2 = nn.Linear(10, 5)

  def forward(self, x):
    y = self.net1(x)
    y = self.relu(y)
    return self.net2(y)


def demo_basic(rank):
  torch.manual_seed(42)

  model = ToyModel()
  # Requires commenting out some assertions inside pytorch since our PG doesn't
  # take CPU tensors, nor do we register the backend for any meaningful device
  # key.
  ddp_model = DDP(model)
  print("state_dict", ddp_model.state_dict())
  ddp_model = torch.compile(ddp_model, backend=my_backend)
  print("post_compile", ddp_model)

  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(
    ddp_model.parameters(), lr=1
  )  # use huge LR to ensure significant change to params
  print([p.mean() for p in ddp_model.parameters()])

  x = torch.randn(20, 10)

  optimizer.zero_grad()
  outputs = ddp_model(x)
  labels = torch.randn(20, 5)
  loss_fn(outputs, labels).backward()
  optimizer.step()

  print([p.mean() for p in ddp_model.parameters()])


class MyInterpreter(torch.fx.Interpreter):
  """Interpreter that just calls dispatch to handle tensors."""

  def __init__(self, graph_module, env=None):
    super().__init__(graph_module)
    self._tx2_env = env or torch_xla2.default_env()

  def call_function(self, target, args, kwargs):
    if not isinstance(
      target, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
    ):
      return super().call_function(target, args, kwargs)

    print("Running ", target, "--------")
    print(args, kwargs)
    return self._tx2_env.dispatch(target, ..., args, kwargs)

  def run_node(self, n):
    res = super().run_node(n)
    return res


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  gm.graph.print_tabular()

  env = torch_xla2.default_env()
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  mesh = Mesh(devices, ("torch_dist"))

  # HACK: shard map inside of torch.compile since compiled function cannot take
  # XLA tensors as input. This is probably not going to work.
  # This currently returns an incorrect shape with multiple copies of parameters
  # and/or gradients. This situation is probably hack-able if the dynamo graph
  # always ends with an allreduce so we can assume outputs are replicated.
  @functools.partial(
    shard_map, mesh=mesh, in_specs=P("torch_dist"), out_specs=P("torch_dist")
  )
  def jax_wrapper(jax_args):
    args = env.j2t_iso(jax_args)
    torch_outputs = MyInterpreter(gm).run(*args)
    print(type(torch_outputs), torch_outputs)
    return env.t2j_iso(list(torch_outputs))

  def f(*torchtensor):
    def convert(t):
      # HACK: replicate so parameters are on each device. I think the right way
      # to "replicate" params is to close on them without taking them as
      # arguments
      if isinstance(t, torch.nn.parameter.Parameter):
        repl = np.repeat(t.numpy(), jax.device_count(), axis=0)
        return repl
      return t.numpy()

    # Make all of the inputs numpy so JAX understands them
    jaxtensors = tuple(convert(t) for t in torchtensor)
    res = jax_wrapper(jaxtensors)
    # Make this a plain CPU tensor again so PyTroch doesn't see XLA tensors
    res_cpu = [
      torch.tensor(r.numpy()) if r is not None else None
      for r in torch_xla2.default_env().j2t_iso(res)
    ]

    return res_cpu

  # box so PyTorch doesn't complain
  return make_boxed_func(f)


my_backend = aot_autograd(fw_compiler=my_compiler)


@torch.compile(backend=my_backend)
def cc(t):
  dist.all_reduce(t)
  return t


if __name__ == "__main__":
  replicas = 4
  # Note: set this before running this script
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={replicas}"

  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group("jax", init_method="jax://")

  # A basic collective works fine, shockingly
  x = torch.ones(4)
  print(cc(x))

  # Models are trickier...
  print(demo_basic(0))
