"""`torch.distributed` backend implemented with JAX collective ops.

EXPERIMENTAL: This module is still highly experimental, and it may be removed
before any stable release.

Note: JAX collective ops require that axis names be defined in `pmap` or
`shmap`. The distributed backend only supports one axis, named `torch_dist`.
This name is defined by our mirror implementation of `spawn`.
"""

import datetime
import functools
import logging
import os
from typing import List, Optional, Union

import jax
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives
from torch._C._distributed_c10d import ProcessGroup  # type: ignore
import torch.distributed
import torch_xla2
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import torch.utils._pytree as torch_pytree
from torch_xla2 import interop


class ProcessGroupJax(ProcessGroup):
  """Distributed backend implemented with JAX."""

  def __init__(self, prefix_store, rank, size, timeout):
    super().__init__(rank, size)
    self._group_name = None

  def getBackendName(self):
    return "jax"

  # TODO(wcromar): why doesn't default group name setter work?
  # https://github.com/pytorch/pytorch/blob/7b1988f9222f3dec5cc2012afce84218199748ae/torch/csrc/distributed/c10d/ProcessGroup.cpp#L148-L152
  def _set_group_name(self, name: str) -> None:
    self._group_name = name

  @property
  def group_name(self):
    assert self._group_name
    return self._group_name

  @staticmethod
  def _work(
    tensors: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
  ) -> dist.Work:
    fut = torch.futures.Future()
    fut.set_result(tensors)
    return torch._C._distributed_c10d._create_work_from_future(fut)

  def _allgather_base(
    self,
    output: torch.Tensor,
    input: torch.Tensor,
    opts=...,
  ) -> dist.Work:
    assert isinstance(input, torch_xla2.tensor.XLATensor2)
    assert isinstance(output, torch_xla2.tensor.XLATensor2)
    torch.distributed._functional_collectives.all_gather_tensor_inplace(
      output, input, group=self
    )
    return self._work(output)

  def allreduce(
    self,
    tensors: List[torch.Tensor],
    opts: dist.AllreduceOptions = ...,
  ) -> dist.Work:
    assert len(tensors) == 1
    assert isinstance(tensors[0], torch_xla2.tensor.XLATensor2)
    torch.distributed._functional_collectives.all_reduce_inplace(
      tensors[0],
      torch.distributed._functional_collectives.REDUCE_OP_TO_STR[
        opts.reduceOp.op
      ],
      self,
    )

    return self._work(tensors)

  def broadcast(
    self,
    tensors: List[torch.Tensor],
    opts: dist.BroadcastOptions = ...,
  ) -> dist.Work:
    assert len(tensors) == 1
    assert isinstance(tensors[0], torch_xla2.tensor.XLATensor2)
    tensors[0].copy_(
      torch.distributed._functional_collectives.broadcast(
        tensors[0], opts.rootRank, group=self
      )
    )

    return self._work(tensors)


dist.Backend.register_backend("jax", ProcessGroupJax)


def jax_rendezvous_handler(
  url: str, timeout: datetime.timedelta = ..., **kwargs
):
  """Initialize distributed store with JAX process IDs.

  Requires `$MASTER_ADDR` and `$MASTER_PORT`.
  """
  # TODO(wcromar): jax.distributed.initialize(...) for multiprocess on GPU
  # TODO(wcromar): Can we use the XLA coordinator as a Store? This isn't part
  # of their public Python API
  master_ip = os.environ["MASTER_ADDR"]
  master_port = int(os.environ["MASTER_PORT"])
  # TODO(wcromar): Use `torchrun`'s store if available
  store = dist.TCPStore(
    master_ip,
    master_port,
    jax.process_count(),
    is_master=jax.process_index() == 0,
  )

  yield (store, jax.process_index(), jax.process_count())


dist.register_rendezvous_handler("jax", jax_rendezvous_handler)


def spawn(f, args=(), env: Optional[torch_xla2.tensor.Environment] = None):
  """Wrap `f` in a JAX `pmap` with the axis name `torch_dist` defined.
  `f` is expected to take the replica index as a positional argument, similar
  to `torch.multiprocessing.spawn`.
  Note: `spawn` does not actually create parallel processes.
  """
  env = env or torch_xla2.default_env()

  def jax_wrapper(index, jax_args):
    index, args = env.j2t_iso([index, jax_args])
    torch_outputs = f(index, *args)
    return env.t2j_iso(torch_outputs)

  jax_outputs = jax.pmap(jax_wrapper, axis_name="torch_dist")(
    np.arange(jax.device_count()), env.t2j_iso(args)
  )
  return env.j2t_iso(jax_outputs)


class DistributedDataParallel(torch.nn.Module):
  """Re-implementation of DistributedDataParallel using JAX SPMD.

  Splits inputs along batch dimension (assumed to be 0) across all devices in
  JAX runtime, including remote devices. Each process should load a distinct
  shard of the input data using e.g. DistributedSampler. Each process' shard
  is then further split among the addressable devices (e.g. local TPU chips)
  by `shard_input`.

  Note: since parameters are replicated across addressable devices, inputs
  must also be SPMD sharded using `shard_input` or `replicate_input`.

  Example usage:

  ```
  jax_model = torch_xla2.distributed.DistributedDataParallel(create_model())
  for data, dataloader:
    jax_data = jax_model.shard_input(data)
    jax_output = jax_model(jax_data)
  ```
  """
  def __init__(
    self,
    module: torch.nn.Module,
    env: Optional[torch_xla2.tensor.Environment] = None,
    **kwargs,
  ):
    if kwargs:
      logging.warning(f"Unsupported kwargs {kwargs}")

    super().__init__()
    self._env = env or torch_xla2.default_env()
    self._mesh = Mesh(
      mesh_utils.create_device_mesh((jax.device_count(),)),
      axis_names=("batch",),
    )
    replicated_state = torch_pytree.tree_map_only(
      torch.Tensor,
      lambda t: self._env.j2t_iso(
        jax.device_put(
          self._env.to_xla(t)._elem, NamedSharding(self._mesh, P())
        )
      ),
      module.state_dict(),
    )
    # TODO: broadcast
    module.load_state_dict(replicated_state, assign=True)
    self._module = module

  def shard_input(self, inp):
    per_process_batch_size = inp.shape[0]  # assumes batch dim is 0
    per_replica_batch_size = per_process_batch_size // jax.local_device_count()
    per_replica_batches = torch.chunk(inp, jax.local_device_count())
    global_batch_size = per_replica_batch_size * jax.device_count()
    global_batch_shape = (global_batch_size,) + inp.shape[1:]

    sharding = NamedSharding(self._mesh, P("batch"))
    return jax.make_array_from_single_device_arrays(
      global_batch_shape,
      NamedSharding(self._mesh, P("batch")),
      arrays=[
        jax.device_put(self._env.to_xla(batch)._elem, device)
        for batch, device in zip(
          per_replica_batches, sharding.addressable_devices
        )
      ],
    )

  def replicate_input(self, inp):
    return self._env.j2t_iso(
      jax.device_put(inp._elem, NamedSharding(self._mesh, P()))
    )

  def jit_step(self, func):
    @interop.jax_jit
    def _jit_fn(states, args):
      self.load_state_dict(states)
      outputs = func(*args)
      return self.state_dict(), outputs

    @functools.wraps(func)
    def inner(*args):
      jax_states = self.state_dict()
      new_states, outputs = _jit_fn(jax_states, args)
      self.load_state_dict(new_states)
      return outputs

    return inner

  def forward(self, *args):
    with self._env:
      return self._module(*args)
