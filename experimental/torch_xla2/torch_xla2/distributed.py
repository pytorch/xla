import datetime
import os
from typing import List, Optional

import jax
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup
import torch_xla2
import numpy as np


class ProcessGroupJax(ProcessGroup):
  def __init__(self, prefix_store, rank, size, timeout):
    super().__init__(rank, size)
    self.prefix_store = prefix_store  # reserved for future use.
    self.timeout = timeout

  def getBackendName(self):
    return "jax"

  def allreduce(
    self,
    tensors: List[torch.Tensor],
    opts: dist.AllreduceOptions = ...,
  ):
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


dist.Backend.register_backend("jax", ProcessGroupJax)


def jax_rendezvous_handler(
  url: str, timeout: datetime.timedelta = ..., **kwargs
):
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
  env = env or torch_xla2.default_env()

  def jax_wrapper(index, jax_args):
    index, args = env.j2t_iso([index, jax_args])
    torch_outputs = f(index, *args)
    return env.t2j_iso(torch_outputs)

  jax_outputs = jax.pmap(jax_wrapper, axis_name="torch_dist")(
    np.arange(jax.device_count()), env.t2j_iso(args)
  )
  return env.j2t_iso(jax_outputs)
