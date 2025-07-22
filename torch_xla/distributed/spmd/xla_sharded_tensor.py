import torch
from torch.utils._pytree import tree_map
import torch_xla

from dataclasses import dataclass
from typing import List, Tuple, Iterator, Union
import contextlib
import collections
import torch_xla.runtime as xr
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard, Replicate
from torch.utils._pytree import tree_map_only


@dataclass
class XLAShard:
  # A snapshot of the shard data from the time of XLAShard creation.
  data: torch.Tensor

  # The indices of the shard into the global tensor. If the tensor is replicated
  # across local devices, the value of `indices` is Ellipsis. Otherwise, it is a
  # list of the index slices across each dimension.
  # The indices do not reflect padding, since the padding does not exist on the
  # global tensor.
  indices: Union[type(Ellipsis), List[slice]]

  # The device this shard's data originated from.
  shard_device: str

  # The replica this shard belongs to, as determined by the sharding. The
  # replica is determined differently for each sharding type:
  #  - TILED:       Since the tensor isn't replicated, replica_id is always 0.
  #  - PARTIAL:     replica_id is taken from the OpSharding and is a value in
  #                 the range [0, num_replica).
  #  - REPLICATED:  Since the tensor is fully replicated, replica_id is the
  #                 device's global ordinal.
  replica_id: int

  @property
  def unpadded_data(self) -> torch.Tensor:
    ''' Returns a copy of `data` with padding removed '''
    unpadded_indices = self.indices
    # Replicated data has Ellipsis as indices
    if self.indices != Ellipsis:
      unpadded_indices = [slice(0, s.stop - s.start) for s in self.indices]
    return self.data[unpadded_indices]

  @unpadded_data.setter
  def unpadded_data(self, t: torch.Tensor):
    unpadded_indices = self.indices
    if self.indices != Ellipsis:
      unpadded_indices = [slice(0, s.stop - s.start) for s in self.indices]
    self.data[unpadded_indices] = t


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

  __slots__ = ['global_tensor', 'mesh_shape', 'partition_spec', '_cached_spec']

  @staticmethod
  def __new__(cls,
              elem: torch.Tensor,
              mesh_shape=None,
              partition_spec=None,
              *args,
              **kwargs):
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

    # Initialize mesh, partition, and spec information
    r.mesh_shape = mesh_shape or (elem.mesh_shape if isinstance(
        elem, XLAShardedTensor) else None)
    r.partition_spec = partition_spec or (elem.partition_spec if isinstance(
        elem, XLAShardedTensor) else None)
    r._cached_spec = None
    return r

  # Shards on the devices are materialized/available after the lazy
  # execution of the partitioned HLO graph. Each XLAShard points
  # to torch.Tensor. The shards represent a snapshot on CPU, detached
  # from the global tensor. The shard data will contain any padding
  # which results from the sharding.
  @property
  def local_shards(self) -> List[XLAShard]:
    shard_dev = torch_xla._XLAC._get_local_shards([self.global_tensor])[0]
    replica_ind = torch_xla._XLAC._get_local_shard_replica_and_indices(
        [self.global_tensor])[0]
    return [
        XLAShard(data, indices, dev, replica)
        for (data, dev), (replica, indices) in zip(shard_dev, replica_ind)
    ]

  # Load the given list of local shards into the underlying tensor's data
  # on the local devices.
  def load_local_shards_(self, shards: List[XLAShard]):
    data = [s.data for s in shards]
    devices = [s.shard_device for s in shards]
    torch_xla._XLAC._load_local_shards(self.global_tensor, data, devices)

    # Invalidate cached spec since the global_tensor data has changed
    self.invalidate_spec_cache()

  @property
  def sharding_spec(self):
    return torch_xla._XLAC._get_xla_sharding_spec(self.global_tensor)

  @property
  def sharding_type(self) -> 'ShardingType':
    from torch_xla.distributed.spmd import ShardingType
    sharding_type = torch_xla._XLAC._get_xla_sharding_type(self.global_tensor)
    return ShardingType(sharding_type)

  def __repr__(self):
    if not hasattr(self, "global_tensor"):
      # materialize a copy of sharded global_tensnor and keep the actual data
      # sharded on the XLA devices.
      return str(self.cpu())
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

  @property
  def _spec(self):
    """
    Convert XLA sharding information to DTensorSpec for DTensor interface compatibility.
    """
    # Return cached spec if available
    if self._cached_spec is not None:
      return self._cached_spec

    # use existing mesh_shape
    if self.mesh_shape is not None:
      device_count = xr.global_runtime_device_count()
      device_list = list(range(device_count))
      mesh = DeviceMesh("xla",
                        torch.tensor(device_list).reshape(self.mesh_shape))
    else:
      raise ValueError(
          "mesh_shape must be specified to create DTensorSpec. "
          "If this tensor was created through torch operations, it may be auto-wrapped. "
          "Use wrap_as_sharded_tensor() to set mesh_shape before accessing _spec. "
      )

    # use existing partition_spec
    if self.partition_spec is not None:
      placements = []
      for mesh_dim in range(len(self.mesh_shape)):
        # find tensor dimension sharded on this mesh dimension
        tensor_dim = None
        for t_dim, m_dim in enumerate(self.partition_spec):
          if m_dim == mesh_dim:
            tensor_dim = t_dim
            break
        placements.append(
            Shard(tensor_dim) if tensor_dim is not None else Replicate())
    else:
      raise ValueError(
          "partition_spec must be specified to create DTensorSpec. "
          "If this tensor was created through torch operations, it may be auto-wrapped. "
          "Use wrap_as_sharded_tensor() to set partition_spec before accessing _spec. "
      )

    # tensor metadata
    tensor_meta = TensorMeta(
        shape=self.global_tensor.shape,
        stride=self.global_tensor.stride(),
        dtype=self.global_tensor.dtype)

    # Create and cache the spec
    self._cached_spec = DTensorSpec(
        mesh=mesh, placements=tuple(placements), tensor_meta=tensor_meta)
    return self._cached_spec

  def invalidate_spec_cache(self):
    """Invalidate the cached DTensorSpec."""
    self._cached_spec = None

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    return super().__torch_function__(func, types, args, kwargs)
