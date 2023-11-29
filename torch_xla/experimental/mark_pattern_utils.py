import uuid
from typing import Dict, Union

import torch
from torch_xla.experimental import xla_marker


class StableHLOCompositeBuilder:

  def __init__(self,
               name: str,
               attrs: Dict[str, Union[int, float, str]] = None):

    self.attrs = attrs
    self.name = name
    self.id = uuid.uuid4().hex
    self._inputs = []
    self._outputs = []

  def _mark_tensor(self, *tensors: torch.Tensor, mark_input=True):
    inputs = []
    for pos, tensor in enumerate(tensors):
      if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"input must be a torch tensor. Got {type(tensor)}.")
      tensor = torch.ops.xla_pattern_marking.mark_tensor(
          tensor,
          name=self.name,
          pos=pos,
          id=self.id,
          is_input=mark_input,
          attr=self.attrs if not mark_input else None)
      inputs.append(tensor)
    if len(inputs) == 1:
      return inputs[0]
    return tuple(inputs)

  def mark_inputs(self, *tensors: torch.Tensor):
    return self._mark_tensor(*tensors, mark_input=True)

  def mark_outputs(self, *tensors: torch.Tensor):
    if len(tensors) > 1:
      # TODO: Allow multiple composite outputs
      raise ValueError(
          f"StableHLO composite with more than one outputs is not supported.")
    return self._mark_tensor(*tensors, mark_input=False)
