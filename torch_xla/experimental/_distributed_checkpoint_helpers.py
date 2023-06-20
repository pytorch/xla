# TODO(jonbolin): These are copies of some upstream APIs which are not yet
# stable. Once the upstream makes these stable, we should take a dependency on
# their APIs.

import torch

from torch.distributed.checkpoint.planner import SavePlan
from typing import (
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,)

PATH_ITEM = Union[str, int]
OBJ_PATH = Tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = Dict[str, OBJ_PATH]

STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]


def _keep_visiting_tensors(value: STATE_DICT_ITEM) -> bool:
  return isinstance(value, torch.Tensor)


def _traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
  """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Traversal is short-circuited when if finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """

  # a value is terminal if it has no other containers values inside it
  def _is_terminal(value: STATE_DICT_ITEM) -> bool:
    values: Collection[STATE_DICT_ITEM]
    if isinstance(value, Mapping):
      values = value.values()
    elif isinstance(value, list):
      values = value
    else:
      return True

    for entry in values:
      if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
        return False
      if keep_traversing is not None and keep_traversing(entry):
        return False
    return True

  def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
    if _is_terminal(value):
      visitor(path, value)
    elif isinstance(value, Mapping):
      for k, v in value.items():
        _traverse_obj(path + (str(k),), v)
    elif isinstance(value, list):
      for i, v in enumerate(value):
        _traverse_obj(path + (i,), v)

  for key, value in state_dict.items():
    _traverse_obj((str(key),), value)


# TODO(jonbolin): Take a dependency on the upstream implementation when the APIs
# are stable
# https://github.com/pytorch/pytorch/blob/d1cecd9c32ba700c27f2b0716bf2cbef41469495/torch/distributed/checkpoint/_traverse.py#L80
def set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH,
                value: STATE_DICT_ITEM) -> None:
  """
    Set ``value`` in ``root_dict`` along the ``path`` object path.
    """
  cur_container = cast(CONTAINER_TYPE, root_dict)

  def extend_list(lst: List[STATE_DICT_ITEM], idx: int) -> None:
    while len(lst) <= idx:
      lst.append(None)

  for i in range(1, len(path)):
    prev_key = path[i - 1]
    key = path[i]
    def_val = cast(STATE_DICT_ITEM, {} if type(key) == str else [])

    if isinstance(cur_container, Mapping):
      cur_container = cast(CONTAINER_TYPE,
                           cur_container.setdefault(prev_key, def_val))
    else:
      extend_list(cur_container, prev_key)
      if cur_container[prev_key] is None:
        cur_container[prev_key] = def_val
      cur_container = cur_container[prev_key]

  key = path[-1]
  if type(key) == int:
    extend_list(cast(List[STATE_DICT_ITEM], cur_container), key)

  cur_container[key] = value


# TODO(jonbolin): Take a dependency on the upstream implementation when the APIs
# are stable
# https://github.com/pytorch/pytorch/blob/d1cecd9c32ba700c27f2b0716bf2cbef41469495/torch/distributed/checkpoint/_nested_dict.py#L27
def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
  """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.
    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
  flattened: STATE_DICT_TYPE = {}
  mappings: FLATTEN_MAPPING = {}

  def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
    new_fqn = ".".join(map(str, path))
    if new_fqn in flattened:
      raise ValueError(f"duplicated flatten key {new_fqn}")
    flattened[new_fqn] = value
    mappings[new_fqn] = path

  _traverse_state_dict(state_dict, flat_copy)
  return flattened, mappings


# TODO(jonbolin): Take a dependency on the upstream implementation when the APIs
# are stable.
# https://github.com/pytorch/pytorch/blob/d1cecd9c32ba700c27f2b0716bf2cbef41469495/torch/distributed/checkpoint/_dedup_tensors.py#L29
def dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
  all_plans = list(all_plans)
  key_to_plan: Dict[MetadataIndex, List[int]] = {}
  for plan_idx, plan in enumerate(all_plans):
    for write_item in plan.items:
      key_to_plan.setdefault(write_item.index, []).append(plan_idx)

  replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

  # Remove duplicates by always keeping the first entry.
  # Compute the per-rank remove set.
  plan_to_keys: Dict[int, List[MetadataIndex]] = {}
  for key, plans in replicated_items.items():
    for plan_idx in plans[1:]:
      plan_to_keys.setdefault(plan_idx, []).append(key)

  for plan_idx, keys in plan_to_keys.items():
    key_set = set(keys)
    # rewrite items and remove elements
    new_items = [
        write_item for write_item in all_plans[plan_idx].items
        if write_item.index not in key_set
    ]
    all_plans[plan_idx] = dataclasses.replace(
        all_plans[plan_idx], items=new_items)

  return all_plans


# TODO(jonbolin): Take a dependency on the upstream implementation when the APIs
# are stable
# https://github.com/pytorch/pytorch/blob/d1cecd9c32ba700c27f2b0716bf2cbef41469495/torch/distributed/_shard/_utils.py#L7
def narrow_tensor_by_index(tensor: torch.Tensor, offsets: Sequence[int],
                           sizes: Sequence[int]) -> torch.Tensor:
  """
    Narrow the tensor according to `offsets` and `sizes`.
    """
  narrowed_tensor = tensor
  for idx, (offset, size) in enumerate(zip(offsets, sizes)):
    if size < tensor.size(idx):
      # Reshape to get shard for this rank and we don't want autograd
      # recording here for the narrow op and 'local_shard' should be a
      # leaf variable in the autograd graph.
      narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
  return narrowed_tensor