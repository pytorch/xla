import dataclasses
import io
import numpy as np
import torch
import torch_xla
import torch_xla.experimental.xla_sharding as xs

from collections import ChainMap
from torch.distributed.checkpoint.default_planner import (
    create_default_local_load_plan,
    create_default_global_load_plan,
    create_default_local_save_plan,
    create_default_global_save_plan,
)
from torch.distributed.checkpoint.planner import (
    SavePlanner,
    LoadPlanner,
    SavePlan,
    LoadPlan,
    ReadItem,
    WriteItem,
    WriteItemType,
    TensorProperties,
    TensorWriteData,
)
from torch.distributed.checkpoint.planner_helpers import (
    create_read_items_for_chunk_list,)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    Metadata,
    STATE_DICT_TYPE,
)
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.utils._pytree import tree_map
from torch_xla.experimental.xla_sharding import (XLAShardedTensor, XLAShard,
                                                 ShardingType)
from torch_xla.experimental._distributed_checkpoint_helpers import (
    FLATTEN_MAPPING,
    flatten_state_dict,
    dedup_tensors,
    set_element,
    narrow_tensor_by_index,
)
from typing import Any, Dict, List, Tuple, Union

__all__ = [
    "SPMDSavePlanner",
    "SPMDLoadPlanner",
]


class SPMDSavePlanner(SavePlanner):
  """
  SPMDSavePlanner provides an implementation of the SavePlanner interface
  which handles state_dicts containing XLAShardedTensor.

  This implementation is based on the DefaultSavePlanner from
  https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/default_planner.py
  """

  def __init__(self):
    # Whether this host is the checkpoint coordinator
    self.is_coordinator: bool = False

    # Mappings created after flattening the state_dict
    self.mappings: FLATTEN_MAPPING = None

    # Flattened state_dict tracking all sharded tensors to be checkpointed
    self.sharded_state_dict: Dict[str, XLAShardedTensor] = None

    # Flattend state_dict tracking all other state_dict items
    self.unsharded_state_dict: Dict[str, Any] = None

    # Upon the first `resolve_data` call for a WriteItem associated with a
    # sharded tensor, all local shards are moved to CPU via
    # `XLAShardedTensor::local_shards` and are tracked in `_local_shards` until
    # the shard's data is resolved. This allows only transferring the shards
    # to CPU once.
    self._local_shards: Dict[str, List[XLAShard]] = {}

  def set_up_planner(self, state_dict: STATE_DICT_TYPE,
                     is_coordinator: bool) -> None:
    self.is_coordinator = is_coordinator

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

  def create_local_plan(self) -> SavePlan:
    # Create the save plan for unsharded data
    plan = create_default_local_save_plan(self.unsharded_state_dict,
                                          self.is_coordinator)
    # Track the flattened mappings in the plan metadata
    plan = dataclasses.replace(plan, planner_data=self.mappings)

    # Extend the plan for sharded tensor data
    xla_write_items = _create_xla_write_items(self.sharded_state_dict)
    plan.items.extend(xla_write_items)
    return plan

  def create_global_plan(
      self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    # Deduplicate write items across plans
    all_plans = dedup_tensors(all_plans)

    global_plan, metadata = create_default_global_save_plan(all_plans)

    # Combine mappings from all plans
    planner_data_dict = [p.planner_data for p in global_plan]
    merged_mappings = dict(ChainMap(*planner_data_dict))
    metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

    return global_plan, metadata

  def finish_plan(self, new_plan: SavePlan) -> SavePlan:
    return new_plan

  def resolve_data(self,
                   write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
    obj = self.lookup_object(write_item.index)
    return self.transform_object(write_item, obj)

  def lookup_object(self, index: MetadataIndex) -> Any:
    if index.fqn in self.unsharded_state_dict:
      return self.unsharded_state_dict[index.fqn]

    if index.fqn not in self._local_shards:
      xtensor = self.sharded_state_dict[index.fqn]
      assert isinstance(xtensor,
                        XLAShardedTensor), f"Unsupported object type: {xtensor}"
      self._local_shards[index.fqn] = xtensor.local_shards

    shard = self._local_shards[index.fqn][index.index]
    assert shard is not None, f"WriteItem has already been processed: {index}"
    assert index.offset == torch.Size(
        ind.start for ind in shard.indices
    ), "WriteItem does not correspond to the correct shard"
    # Release the local shard
    self._local_shards[index.fqn][index.index] = None
    return shard.unpadded_data

  def transform_object(self, write_item: WriteItem, object: Any):
    if write_item.type == WriteItemType.BYTE_IO:
      bytes = io.BytesIO()
      torch.save(object, bytes)
      object = bytes
    return object


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


def _create_write_item_from_indices(fqn: str, shard_index: int,
                                    indices: List[slice],
                                    global_size: torch.Size,
                                    properties: TensorProperties) -> WriteItem:
  offsets = torch.Size(ind.start for ind in indices)
  sizes = torch.Size(ind.stop - ind.start for ind in indices)
  return WriteItem(
      index=MetadataIndex(fqn, offsets, shard_index),
      type=WriteItemType.SHARD,
      tensor_data=TensorWriteData(
          chunk=ChunkStorageMetadata(
              offsets=offsets,
              sizes=sizes,
          ),
          properties=properties,
          size=global_size,
      ),
  )


def _create_write_items_for_xla_sharded_tensor(
    fqn: str, t: XLAShardedTensor) -> List[WriteItem]:
  items = []
  # Since local shards are currently moved to CPU on creation, we need to get
  # the shard indices indirectly to avoid unnecessarily consuming host memory.
  shard_indices = torch_xla._XLAC._get_local_shard_indices(t.global_tensor)
  prop = TensorProperties.create_from_tensor(t)
  for shard_ind, indices in enumerate(shard_indices):
    write_item = _create_write_item_from_indices(fqn, shard_ind, indices,
                                                 t.size(), prop)
    items.append(write_item)
  return items


def _create_xla_write_items(state_dict: STATE_DICT_TYPE) -> List[WriteItem]:
  """
  Iterate through the state_dict and return WriteItems for all local shards
  """
  items = []
  for fqn, v in state_dict.items():
    assert isinstance(v, XLAShardedTensor
                     ), '_create_xla_write_items only accepts XLAShardedTensor'
    items.extend(_create_write_items_for_xla_sharded_tensor(fqn, v))
  return items


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
