import dataclasses
import json
from dataclasses import dataclass
from typing import Dict

import torch
import torch_xla
from torch.library import Library, impl

xla_pattern_marking_lib = Library("xla_pattern_marking", "DEF")

xla_pattern_marking_lib.define(
    "mark_tensor(Tensor x, str name, int pos, str id, bool is_input, Any? attr=None) -> Tensor"
)


@dataclass
class BoundaryMetadata:
  name: str  # Name of the Patttern.
  pos: int  # Arg/return position.
  id: str  # Patten instance id.
  is_input: bool = True  # If the marked tensor is input/output.
  attr: dict = None  # Attribute of the pattern, expected to be attached to output.


class BoundaryMetadataSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj)
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
                    id: str,
                    is_input: bool,
                    attr: Dict = None):
  """Attach pattern boundary metadata to a XLA Tensor.
  
  Args:
      x: torch.Tensor (On XLA device) - the marked tensor.
      name: str - The name of the pattern, it will be the name of the stablehlo composite op.
      pos: int - Input/output Position of the annotated tensor in the pattern.
      id: str - Unique identifier of the pattern instance.
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
                id: str,
                is_input: bool,
                attr: Dict = None):
  # Do nothing for non-xla tensor.
  return x


@impl(xla_pattern_marking_lib, "mark_tensor", "Meta")
def mark_tensor_meta(x: torch.Tensor,
                     name: str,
                     pos: int,
                     id: str,
                     is_input: bool,
                     attr: Dict = None):
  return torch.empty_like(x)
