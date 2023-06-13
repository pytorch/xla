import io
import numpy as np
import torch
import torch_xla
import torch_xla.experimental.xla_sharding as xs

from torch.distributed.checkpoint.default_planner import (
    create_default_local_load_plan,
    create_default_global_load_plan,
)
from torch.distributed.checkpoint.planner import (
    SavePlanner,
    LoadPlanner,
    SavePlan,
    LoadPlan,
    ReadItem,
    WriteItem,
)
from torch.distributed.checkpoint.planner_helpers import (
    create_read_items_for_chunk_list,)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    Metadata,
    STATE_DICT_TYPE,
)
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
    flatten_state_dict,
)
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.utils._pytree import tree_map
from torch_xla.experimental.xla_sharding import (XLAShardedTensor, XLAShard,
                                                 ShardingType)
from typing import Any, Dict, List, Tuple, Union

__all__ = [
    "SPMDSavePlanner",
    "SPMDLoadPlanner",
]


class SPMDSavePlanner(SavePlanner):
  """
  SPMDSavePlanner provides an implementation of the SavePlanner interface
  which handles state_dicts containing XLAShardedTensor or List[XLAShard].
  """

  def set_up_planner(self, state_dict: STATE_DICT_TYPE,
                     is_coordinator: bool) -> None:
    raise NotImplemented

  def create_local_plan(self) -> SavePlan:
    raise NotImplemented

  def create_global_plan(
      self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    raise NotImplemented

  def finish_plan(self, new_plan: SavePlan) -> SavePlan:
    raise NotImplemented

  def resolve_data(self,
                   write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
    raise NotImplemented


class SPMDLoadPlanner(LoadPlanner):
  """
  SPMDLoadPlanner provides an implementation of the LoadPlanner interface
  which handles state_dicts containing XLAShardedTensor.

  The input state_dict should already be sharded and on the XLA device, and
  tensors and shards will be loaded in-place.

  This implementation is based on the DefaultLoadPlanner from
  https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/default_planner.py
  """

  def __init__(self):
    # Checkpoint metadata
    self.metadata: Metadata = None

    # Whether this host is the checkpoint coordinator
    self.is_coordinator: bool = False

    # The original state_dict passed to `set_up_planner`
    self.original_state_dict: STATE_DICT_TYPE = None

    # Mappings created after flattening the state_dict
    self.mappings: FLATTEN_MAPPING = None

    # Flattened state_dict tracking all sharded tensors to be restored
    self.sharded_state_dict: Dict[str, XLAShardedTensor] = None

    # Flattend state_dict tracking all other state_dict items
    self.unsharded_state_dict: Dict[str, Any] = None

    # Upon the first `resolve_tensor` call for a ReadItem associated with a
    # sharded tensor, all local shards are moved to CPU via
    # `XLAShardedTensor::local_shards` and are tracked in `_local_shards`.
    # The checkpoint data will be loaded into _local_shards on CPU and
    # moved to the underlying tensor via `XLAShardedTensor::load_local_shards_`
    # when the last shard is fully committed in `commit_tensor`.
    self._local_shards: Dict[str, List[XLAShard]] = {}

    # Track how many tensor elements remain to be read for a sharded tensor.
    self._pending_elements: Dict[str, int] = {}

  def set_up_planner(
      self,
      state_dict: STATE_DICT_TYPE,
      metadata: Metadata,
      is_coordinator: bool,
  ) -> None:
    self.metadata = metadata
    self.is_coordinator = is_coordinator
    self.original_state_dict = state_dict

    # Flatten the state_dict to allow separating sharded XLA tensors from
    # types that can be handled by the default planner, and ensure all sharded
    # tensors are wrapped in XLAShardedTensor
    state_dict, self.mappings = flatten_state_dict(state_dict)
    state_dict = tree_map(xs.wrap_if_sharded, state_dict)

    # Select only XLAShardedTensors which are not replicated, since the
    # default planner can handle everything else.
    self.sharded_state_dict = {
        k: v for k, v in state_dict.items() if _is_sharded_tensor(v)
    }
    unsharded = dict(state_dict.items() - self.sharded_state_dict.items())
    self.unsharded_state_dict = tree_map(_unwrap_xla_sharded_tensor, unsharded)

  def create_local_plan(self) -> LoadPlan:
    # Create the load plan for unsharded data
    plan = create_default_local_load_plan(self.unsharded_state_dict,
                                          self.metadata)
    # Extend the plan for sharded tensor data
    xla_read_items = _create_xla_read_items(self.sharded_state_dict,
                                            self.metadata)
    plan.items.extend(xla_read_items)
    return plan

  def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
    # Processing the local plans to create a global plan does not depend on
    # the underlying tensor types, so we can directly reuse the default
    # planner logic.
    return create_default_global_load_plan(global_plan)

  def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
    return central_plan

  def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
    # Write the value into the original state_dict to load in-place
    set_element(
        self.original_state_dict,
        self.mappings[read_item.dest_index.fqn],
        torch.load(value),
    )

  def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
    tensor = self.lookup_tensor(read_item.dest_index)
    return self.transform_tensor(read_item, tensor)

  def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
    if index.fqn in self.unsharded_state_dict:
      return find_state_dict_object(self.unsharded_state_dict, index)

    if index.fqn not in self._local_shards:
      xtensor = self.sharded_state_dict[index.fqn]
      self._local_shards[index.fqn] = xtensor.local_shards
      # Calculate the expected number of reads for all shards of the tensor
      self._pending_elements[index.fqn] = 0
      for shard in self._local_shards[index.fqn]:
        self._pending_elements[index.fqn] += shard.unpadded_data.numel()

    xla_shard = self._local_shards[index.fqn][index.index]
    assert index.offset == torch.Size(
        ind.start for ind in xla_shard.indices
    ), "ReadItem does not correspond to the correct shard"
    # Return padded data since the tensor will be narrowed to match
    # the ReadItem in `transform_tensor`.
    return xla_shard.data

  def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
    """
    The storage layer requires that the shape of tensors returned by
    `resolve_tensor` matches the `lengths` of the ReadItem. This function
    will return a narrowed view of the tensor that matches the ReadItem's
    lengths and offsets into the global tensor.
    """
    offsets = read_item.dest_offsets
    index = read_item.dest_index
    if index.fqn in self.sharded_state_dict:
      # Update offsets to index into the shard rather than the global tensor
      shard = self._local_shards[index.fqn][index.index]
      offsets = torch.Size(d - i.start for d, i in zip(offsets, shard.indices))
    return narrow_tensor_by_index(tensor, offsets, read_item.lengths)

  def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
    fqn = read_item.dest_index.fqn
    if fqn not in self.sharded_state_dict:
      return

    self._pending_elements[fqn] -= np.prod(read_item.lengths)
    assert self._pending_elements[
        fqn] >= 0, f"Too many writes for tensor {index.fqn}"
    if self._pending_elements[fqn] == 0:
      # Load local shards into the XLAShardedTensor and release the shards
      # from CPU
      local_shards = self._local_shards.pop(fqn)
      self.sharded_state_dict[fqn].load_local_shards_(local_shards)


def _create_chunk_from_shard_index(index: List[slice]) -> ChunkStorageMetadata:
  return ChunkStorageMetadata(
      offsets=torch.Size(ind.start for ind in index),
      sizes=torch.Size(ind.stop - ind.start for ind in index))


def _create_xla_read_items(sharded_state_dict: STATE_DICT_TYPE,
                           metadata: Metadata) -> List[ReadItem]:
  """
  Iterate through the state_dict and return ReadItems for all local shards.
  """
  items = []
  for fqn, t in sharded_state_dict.items():
    assert isinstance(t, XLAShardedTensor
                     ), "_create_xla_read_items only accepts XLAShardedTensor"
    md = metadata.state_dict_metadata[fqn]
    # Since local shards are currently moved to CPU on creation, we need to get
    # the shard indices indirectly to avoid unnecessarily consuming host memory.
    shard_indices = torch_xla._XLAC._get_local_shard_indices(t.global_tensor)
    chunks = [_create_chunk_from_shard_index(index) for index in shard_indices]
    items.extend(create_read_items_for_chunk_list(fqn, md, chunks))
  return items


def _is_sharded_tensor(x: Any) -> bool:
  """Return true if the tensor's data is sharded across multiple devices"""
  return isinstance(
      x, XLAShardedTensor) and x.sharding_type != ShardingType.REPLICATED


def _unwrap_xla_sharded_tensor(x: Any) -> Any:
  if isinstance(x, XLAShardedTensor):
    return x.global_tensor
  return x
