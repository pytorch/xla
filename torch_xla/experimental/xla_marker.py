import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, Union
import uuid

import torch
import torch_xla
from torch.library import Library, impl

xla_pattern_marking_lib = Library("xla_pattern_marking", "DEF")

xla_pattern_marking_lib.define(
    "mark_tensor(Tensor x, str name, int pos, int id, bool is_input, Any? attr=None) -> Tensor"
)

xla_pattern_marking_lib.define(
    "mark_tensor.tensor(Tensor x, str name, int pos, Tensor id, bool is_input, Any? attr=None) -> Tensor"
)


def _get_uuid_tensor_internal(id: uuid.UUID):
  int_arr = []
  for i in range(4):
    int_arr.append(int(id.int >> (128 - 32 * (i + 1)) & 0xFFFFFFFF))
  # Need to use int64 here to avoid an overflow issue in torch.
  return torch.tensor(int_arr, dtype=torch.int64)


def get_uuid_tensor():
  id = uuid.uuid4()
  return _get_uuid_tensor_internal(id)


def decode_uuid_tensor(x):
  assert len(
      x.shape
  ) == 1, f"The uuid tensor is expected to be a 1D tensor. Getting shape : {x.shape}."
  assert x.numel(
  ) == 4, f"The uuid tensor is expected to have 4 elements. Tensor has {x.numel()} elements."
  uuid_int = 0
  for i in range(4):
    uuid_int += x.cpu()[i] << (32 * i)
  return hex(uuid_int)


@dataclass
class BoundaryMetadata:
  name: str  # Name of the Patttern.
  pos: int  # Arg/return position.
  id: Union[int, torch.Tensor]  # Patten instance id.
  is_input: bool = True  # If the marked tensor is input/output.
  attr: dict = None  # Attribute of the pattern, expected to be attached to output.


class BoundaryMetadataSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      if isinstance(obj, BoundaryMetadata):
        if isinstance(obj.id, torch.Tensor):
          obj.id = decode_uuid_tensor(obj.id)
        else:
          obj.id = str(obj.id)
      return dataclasses.asdict(obj)
    else:
      return super().default(obj)


def _assert_valid_composite_attr(attr):
  if attr is None:
    return
  if not isinstance(attr, dict):
    raise ValueError("Composite attr must be a Python dictionary.")

  for k, v in attr.items():
    if not isinstance(k, str):
      raise ValueError("Composite attr name must be a Python str.")
    if type(k) not in [str, float, int]:
      raise ValueError(
          "Composite attr value must be either Python str, float, or int.")


@impl(xla_pattern_marking_lib, "mark_tensor", "XLA")
def mark_tensor_xla(x: torch.Tensor,
                    name: str,
                    pos: int,
                    id: int,
                    is_input: bool,
                    attr: Dict = None):
  """Attach pattern boundary metadata to a XLA Tensor.
  
  Args:
      x: torch.Tensor (On XLA device) - the marked tensor.
      name: str - The name of the pattern, it will be the name of the stablehlo composite op.
      pos: int - Input/output Position of the annotated tensor in the pattern.
      id: int - Unique identifier of the pattern instance.
      is_input: bool - If the annotated tensor is the input to the pattern.
      attr: dict - Attribute of the pattern, it will be passed down to the attribute field
                   in the stablehlo composite.
  """
  _assert_valid_composite_attr(attr)
  pattern_info = BoundaryMetadata(name, pos, id, is_input, attr)
  return torch_xla._XLAC._xla_mark_tensor(
      x, json.dumps(pattern_info, cls=BoundaryMetadataSerializer))


@impl(xla_pattern_marking_lib, "mark_tensor", "CompositeExplicitAutograd")
def mark_tensor(x: torch.Tensor,
                name: str,
                pos: int,
                id: int,
                is_input: bool,
                attr: Dict = None):
  # Do nothing for non-xla tensor.
  return x


@impl(xla_pattern_marking_lib, "mark_tensor", "Meta")
def mark_tensor_meta(x: torch.Tensor,
                     name: str,
                     pos: int,
                     id: int,
                     is_input: bool,
                     attr: Dict = None):
  return torch.empty_like(x)


@impl(xla_pattern_marking_lib, "mark_tensor.tensor", "XLA")
def mark_tensor_xla(x: torch.Tensor,
                    name: str,
                    pos: int,
                    id: torch.Tensor,
                    is_input: bool,
                    attr: Dict = None):
  """Variant: `id` is a torch.Tensor, which is generated from `get_uuid_tensor`.
  """
  _assert_valid_composite_attr(attr)
  pattern_info = BoundaryMetadata(name, pos, id, is_input, attr)
  return torch_xla._XLAC._xla_mark_tensor(
      x, json.dumps(pattern_info, cls=BoundaryMetadataSerializer))


@impl(xla_pattern_marking_lib, "mark_tensor.tensor",
      "CompositeExplicitAutograd")
def mark_tensor(x: torch.Tensor,
                name: str,
                pos: int,
                id: torch.Tensor,
                is_input: bool,
                attr: Dict = None):
  # Do nothing for non-xla tensor.
  return x


@impl(xla_pattern_marking_lib, "mark_tensor.tensor", "Meta")
def mark_tensor_meta(x: torch.Tensor,
                     name: str,
                     pos: int,
                     id: torch.Tensor,
                     is_input: bool,
                     attr: Dict = None):
  return torch.empty_like(x)
