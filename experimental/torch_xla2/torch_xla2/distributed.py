"""`torch.distributed` backend implemented with JAX collective ops.

EXPERIMENTAL: This module is still highly experimental, and it may be removed
before any stable release.

Note: JAX collective ops require that axis names be defined in `pmap` or
`shmap`. The distributed backend only supports one axis, named `torch_dist`.
This name is defined by our mirror implementation of `spawn`.
"""

import datetime
import os
from typing import List, Optional

import jax
import jax.numpy as jnp
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup  # type: ignore
import torch_xla2
import numpy as np


class ProcessGroupJax(ProcessGroup):
  """Distributed backend implemented with JAX."""

  def __init__(self, prefix_store, rank, size, timeout):
    super().__init__(rank, size)

  def getBackendName(self):
    return "jax"

  def allgather(
    self,
    output_tensors: List[List[torch.Tensor]],
    input_tensors: List[torch.Tensor],
    opts = ...
  ) -> dist.Work:
    assert len(input_tensors) == 1, "Only one input tensor supported"
    input_tensor = input_tensors[0]
    assert len(output_tensors) == 1, "Only one input tensor supported"
    output_tensors = output_tensors[0]

    assert isinstance(input_tensor, torch_xla2.tensor.XLATensor2)
    output = jax.lax.all_gather(input_tensor._elem, axis_name="torch_dist")
    output_size = jax.numpy.shape(output)[0]

    assert len(output_tensors) == output_size
    for i, t in enumerate(output_tensors):
      assert isinstance(t, torch_xla2.tensor.XLATensor2)
      t._elem = output[i]

    fut = torch.futures.Future()
    fut.set_result(output_tensors)
    return torch._C._distributed_c10d._create_work_from_future(fut)

  def allreduce(
    self,
    tensors: List[torch.Tensor],
    opts: dist.AllreduceOptions = ...,
  ) -> dist.Work:
    for t in tensors:
      assert isinstance(t, torch_xla2.tensor.XLATensor2)
      match opts.reduceOp:
        case dist.ReduceOp.SUM:
          res = jax.lax.psum(t._elem, axis_name="torch_dist")
        case dist.ReduceOp.AVG:
          res = jax.lax.pmean(t._elem, axis_name="torch_dist")
        case dist.ReduceOp.MIN:
          res = jax.lax.pmin(t._elem, axis_name="torch_dist")
        case dist.ReduceOp.MAX:
          res = jax.lax.pmax(t._elem, axis_name="torch_dist")
        case _:
          raise RuntimeError(f"Reduce op {opts.reduceOp} not implemented")

      t._elem = res

    fut = torch.futures.Future()
    fut.set_result(tensors)
    return torch._C._distributed_c10d._create_work_from_future(fut)

  def broadcast(
    self,
    tensors: List[torch.Tensor],
    opts: dist.BroadcastOptions = ...,
  ) -> dist.Work:
    for t in tensors:
      assert isinstance(t, torch_xla2.tensor.XLATensor2)
      masked = jnp.where(
        jax.lax.axis_index("torch_dist") == opts.rootRank,
        t._elem,
        jnp.zeros_like(t._elem),
      )
      t._elem = jax.lax.psum(masked, "torch_dist")

    fut = torch.futures.Future()
    fut.set_result(tensors)
    return torch._C._distributed_c10d._create_work_from_future(fut)


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


# TODO(wcromar): types
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
