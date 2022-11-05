import torch
from torch.utils._pytree import tree_map
import torch_xla

from dataclasses import dataclass
from typing import List, Tuple, Iterator
import contextlib
import collections


@dataclass
class XLAShard:
  data: torch.Tensor
  rank: int


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
  guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
  try:
    yield
  finally:
    del guard


class XLAShardedTensor(torch.Tensor):
  """
    A wrapper around `torch.Tensor` with sharding annotation
    for XLA SPMD auto-sharding. The wrapped tensors are unwrapped
    for IR tracing and converted to HLO graph with sharding annotations;
    XLA SPMDPartitioner takes a pass, propagating and injecting collectives
    to the graph before compilation.
  """

  # XLAShardedTensor behaves like a unpartitioned,
  # combined tensor on the host machine. When user annotates,
  # this is simply set to the input tensor. When an XLA partitioned
  # output tensor returns (or sharding propagated intermediate tensors)
  # as XLAShardedTensor, the backend gathers global data across devices
  # and materialize and set `global_tensor` on the host; the actual device
  # data still remain on individual device as sharded or replicated.
  # Note: we should drop this reference, and force all gather on each access.
  global_tensor: torch.Tensor
  # Shards on the devices are materialized/available after the lazy
  # execution of the SPMDPartitioned HLO graph; otherwise,
  # local_shards is set to `None`. Each XLAShard points to
  # torch.Tensor (xla::device_data).
  # Note: we can consider returning a callback or even define
  # sharding at XLAShardedTensor construction after pjrt migration.
  local_shards: List[XLAShard] = None
  # A logical device topology, each element describes
  # a number of devices in the corresponding axis.
  # NOTE: we could use more specific device-rank mapping, e.g., ShardingSpec,
  # if needed. The change shouldn't be difficult, or create another constructor.
  mesh_shape: Tuple[int]  # TODO: create a wrapper for named axes
  # Specifies how each input rank is sharded (index to mesh_shape)
  # or replicated (None). For example, we can shard an 8x10 tensor
  # 4-way row-wise, and replicate column-wise.
  # >> input = torch.randn(8, 10)
  # >> mesh_shape = (4, 2)
  # >> assert np.prod(mesh_shape) == len(xm.get_xla_supported_devices())
  # >> partition_spec = (0, None)
  # >> assert len(input.shape) == len(partition_spec)
  partition_spec: Tuple[int, None]

  __slots__ = ['global_tensor']

  @staticmethod
  def __new__(cls, elem: torch.Tensor, *args, **kwargs):
    # TODO(yeounoh) wrapper can take different arguments
    r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
        cls,
        elem.size(),
        strides=elem.stride(),
        storage_offset=elem.storage_offset(),
        dtype=elem.dtype,
        layout=elem.layout,
        device=elem.device,
        requires_grad=kwargs.get("requires_grad", False))
    r.global_tensor = elem.detach() if r.requires_grad else elem
    return r

  @property
  def sharding_spec(self):
    return torch_xla._XLAC._get_xla_sharding_spec(self.global_tensor)

  @property
  def shards(self):
    # Return a list of local shards
    return NotImplemented

  def __repr__(self):
    return f"XLAShardedTensor({self.global_tensor})"

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    """
      The dispatcher allows the unwrapped torch.Tensor to re-dispatched to the
      `xla` backend as XlaTensor, and the XlaTensor with an associated sharding spec
      to be received and wrapped as XLAShardedTensor.
    """

    def unwrap(elem):
      return elem.global_tensor if isinstance(elem, XLAShardedTensor) else elem

    def wrap(elem):
      return XLAShardedTensor(elem) if isinstance(elem, torch.Tensor) else elem

    # no_dispatch is only needed if you use enable_python_mode.
    # It prevents infinite recursion.
    with no_dispatch():
      # re-dispatch to C++
      rs = tree_map(wrap,
                    func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
    return rs
