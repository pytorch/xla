import dataclasses
import json
from dataclasses import dataclass
from typing import Dict

import torch
import torch_xla
from torch.library import Library, impl

xla_pattern_marking_lib = Library("xla_pattern_marking", "DEF")

xla_pattern_marking_lib.define(
    "mark_tensor(Tensor x, str name, int pos, int id, bool is_input, Any? attr=None) -> Tensor"
)


@dataclass
class BoundaryMetadata:
  name: str  # Name of the Patttern.
  pos: int  # Arg/return position.
  id: int  # Patten instance id.
  is_input: bool = True  # If the marked tensor is input/output.
  attr: dict = None  # Attribute of the pattern, expected to be attached to output.


class BoundaryMetadataSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj)
    return super().default(obj)


@impl(xla_pattern_marking_lib, "mark_tensor", "XLA")
def mark_tensor(x: torch.Tensor,
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
def mark_tensor(x: torch.Tensor,
                name: str,
                pos: int,
                id: int,
                is_input: bool,
                attr: Dict = None):
  return torch.empty_like(x)
