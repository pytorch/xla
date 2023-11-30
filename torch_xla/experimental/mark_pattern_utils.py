import uuid
from typing import Dict, Union

import torch
from torch_xla.experimental import xla_marker


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
    self.id = uuid.uuid4().hex
    self._inputs = []
    self._outputs = []

  def _mark_tensor(self, *tensors: torch.Tensor, is_input: bool):
    marked_tensors = []
    for pos, tensor in enumerate(tensors):
      if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"input must be a torch tensor. Got {type(tensor)}.")
      marked_tensors.append(
          torch.ops.xla_pattern_marking.mark_tensor(
              tensor,
              name=self.name,
              pos=pos,
              id=self.id,
              is_input=mark_input,
              attr=self.attr if not is_input else None,
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

    if len(tensors) > 1:
      # TODO: Allow multiple composite outputs
      raise ValueError(
          f"StableHLO composite with more than one outputs is not supported.")
    return self._mark_tensor(*tensors, is_input=False)
