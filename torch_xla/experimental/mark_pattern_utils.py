import uuid
from typing import Dict, Union

import torch
import torch._dynamo as torchdynamo
from torch_xla.experimental import xla_marker


@torchdynamo.assume_constant_result
def _get_uuid() -> str:
  return uuid.uuid4().hex


class StableHLOCompositeBuilder:
  """
  Helper for building a StableHLO Composite by marking input and output tensors. It
  should be used with the StableHLO converters from `torch_xla.stablehlo`.

  Args:
    name (str):
      The name of the built StableHLO Composite op.
    attr (dict):
      Attributes of the StableHLO Composite op.
  """

  def __init__(self, name: str, attr: Dict[str, Union[int, float, str]] = None):

    self.attr = attr
    self.name = name
    self.id = _get_uuid()
    self._inputs = []
    self._outputs = []

  def _mark_tensor(self, *tensors: torch.Tensor, is_input: bool):
    marked_tensors = []
    serialized_attr = xla_marker.serialize_composite_attr(
        self.attr) if not is_input else None

    for pos, tensor in enumerate(tensors):
      if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"input must be a torch tensor. Got {type(tensor)}.")
      marked_tensors.append(
          torch.ops.xla.mark_tensor(
              tensor,
              name=self.name,
              pos=pos,
              id=self.id,
              is_input=is_input,
              attr=serialized_attr,
          ))

    if len(marked_tensors) == 1:
      return marked_tensors[0]
    return tuple(marked_tensors)

  def mark_inputs(self, *tensors: torch.Tensor):
    """
    Mark the input tensors of the StableHLO Composite. This method must only be
    called once per builder.

    Args:
      *tensors (torch.Tensor):
        Torch tensors to mark.
    Returns:
      marked_tensors (torch.Tensor or Tuple[torch.Tensor]):
        Torch tensors marked as composite inputs. The tensor inputs of this method
        should be replaced by the marked tensors in later usages.
    """

    return self._mark_tensor(*tensors, is_input=True)

  def mark_outputs(self, *tensors: torch.Tensor):
    """
    Mark the output tensors of the StableHLO Composite. This method must only be
    called once per builder.

    Args:
      *tensors (torch.Tensor):
        Torch tensors to mark.
    Returns:
      marked_tensors (torch.Tensor or Tuple[torch.Tensor]):
        Torch tensors marked as composite outputs. The tensor inputs of this method
        should be replaced by the marked tensors in later usages.
    """

    return self._mark_tensor(*tensors, is_input=False)
