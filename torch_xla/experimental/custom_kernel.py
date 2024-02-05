import torch
import torch_xla

from typing import List
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "tpu_custom_call_(Tensor(a!) output, Tensor[] inputs, str payload) -> ()",
)

@impl(XLA_LIB, "tpu_custom_call_", "XLA")
def tpu_custom_call_xla_(output: torch.Tensor, inputs: List[torch.Tensor], payload: str):
  torch_xla._XLAC._xla_tpu_custom_call_(output, inputs, payload)


@impl(XLA_LIB, "tpu_custom_call_", "CompositeExplicitAutograd")
def tpu_custom_call_(output: torch.Tensor, inputs: List[torch.Tensor], payload: str):
  # Do nothing for non-xla tensor.
  return
