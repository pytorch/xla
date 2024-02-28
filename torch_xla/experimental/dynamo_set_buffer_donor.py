import torch
import torch_xla

from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB


XLA_LIB.define(
    "dynamo_set_buffer_donor(Tensor t, bool should_donoate) -> Tensor"
)


@impl(XLA_LIB, "dynamo_set_buffer_donor", "XLA")
def dynamo_set_buffer_donor_xla(t: torch.Tensor, should_donoate: bool):
  torch_xla._XLAC._set_buffer_donation(t, should_donoate)
  return t


@impl(XLA_LIB, "dynamo_set_buffer_donor", "CompositeExplicitAutograd")
def dynamo_set_buffer_donor(t: torch.Tensor, should_donoate: bool):
  # Do nothing for non-xla tensor.
  return t