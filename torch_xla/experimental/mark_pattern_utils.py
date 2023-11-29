import uuid
from typing import Dict, Union

import torch
from torch_xla.experimental import xla_marker


def _xla_mark_tensor(*args, **kwargs):
  return torch.ops.xla_pattern_marking.mark_tensor(*args, **kwargs)


class StableHLOCompositeBuilder:

  def __init__(self,
               name: str,
               attrs: Dict[str, Union[int, float, str]] = None):

    self.attrs = attrs
    self.name = str(name)
    self.id = uuid.uuid1().hex
    self._inputs = {}
    self._outputs = {}

  @property
  def uid(self):
    return f"{self.name}__{self.id}"

  def get_input(pos: int) -> torch.Tensor:
    return self._inputs[pos]

  def get_output(pos: int) -> torch.Tensor:
    return self._outputs[pos]

  def mark_input(self, x: torch.Tensor, pos: int) -> torch.Tensor:
    if pos in self._inputs:
      raise ValueError(f"Input {pos} has been marked for {self.uid}.")
    if not isinstance(pos, int) or pos < 0:
      raise ValueError(f"Invalid input position: {pos}.")
    if not isinstance(x, torch.Tensor):
      raise ValueError(f"input must be a torch tensor. Got {type(x)}.")

    x = _xla_mark_tensor(
        x,
        name=self.name,
        pos=pos,
        id=self.id,
        is_input=True,
        attr=self.attrs,
    )
    self._inputs[pos] = x
    return x

  def mark_inputs(self, *tensors: torch.Tensor):
    inputs = tuple(self.mark_input(x, pos=i) for i, x in enumerate(tensors))
    if len(inputs) == 1:
      return inputs[0]
    return inputs

  def mark_output(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
    if pos in self._outputs:
      raise ValueError(f"Output {pos} has been marked for {self.uid}.")
    if not isinstance(pos, int) or pos < 0:
      raise ValueError(f"Invalid output position: {pos}.")
    if not isinstance(x, torch.Tensor):
      raise ValueError(f"Output must be a torch tensor. Got {type(x)}.")
    if pos >= 1:
      # TODO: Allow multiple composite outputs
      raise ValueError(
          f"StableHLO composite with more than one outputs is not supported.")

    x = _xla_mark_tensor(
        x,
        name=self.name,
        pos=pos,
        id=self.id,
        is_input=False,
        attr=self.attrs,
    )
    self._outputs[pos] = x
    return x

  def mark_outputs(self, *tensors: torch.Tensor):
    outputs = tuple(self.mark_output(x, pos=i) for i, x in enumerate(tensors))
    if len(outputs) == 1:
      return outputs[0]
    return outputs
