import torch
import torch_xla

from typing import List, Any, Set
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB
import torch_xla.distributed.spmd as xs

import ast

XLA_LIB.define(
    "dynamo_mark_sharding(Tensor input, int[] device_ids, int[] mesh_shape, str axis_names, str partition_spec) -> Tensor"
)


@impl(XLA_LIB, "dynamo_mark_sharding", "XLA")
def dynamo_mark_sharding_xla(input: torch.Tensor, device_ids: List[int],
                             mesh_shape: List[int], axis_names: str,
                             partition_spec: str):
  mesh_shape_tuple = tuple(mesh_shape)
  axis_names_eval = ast.literal_eval(axis_names)
  mesh = xs.Mesh(device_ids, mesh_shape, axis_names_eval)
  partition_spec_eval = ast.literal_eval(partition_spec)
  return xs.mark_sharding(input, mesh, partition_spec_eval)


@impl(XLA_LIB, "dynamo_mark_sharding", "CompositeExplicitAutograd")
def dynamo_mark_sharding(input: torch.Tensor, device_ids: List[int],
                         mesh_shape: List[int], axis_names: str,
                         partition_spec: str):
  # Do nothing for non-xla tensor.
  return input


XLA_LIB.define(
    "dynamo_set_buffer_donor_(Tensor t, bool should_donoate) -> Tensor")


@impl(XLA_LIB, "dynamo_set_buffer_donor_", "XLA")
def dynamo_set_buffer_donor_xla_(t: torch.Tensor, should_donoate: bool):
  torch_xla._XLAC._set_buffer_donation(t, should_donoate)
  return t


@impl(XLA_LIB, "dynamo_set_buffer_donor_", "CompositeExplicitAutograd")
def dynamo_set_buffer_donor_(t: torch.Tensor, should_donoate: bool):
  # Do nothing for non-xla tensor.
  return t
