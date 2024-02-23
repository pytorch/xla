import torch
import torch_xla

from typing import Tuple, Union, List, Sequence, Any, Optional, Set
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB
import torch_xla.distributed.spmd as xs

import ast

XLA_LIB.define(
    "dynamo_mark_sharding(Tensor t, int[] device_ids, str mesh_shape, str axis_names, str partition_spec) -> Tensor"
)


@impl(XLA_LIB, "dynamo_mark_sharding", "XLA")
def dynamo_mark_sharding_xla(t: torch.Tensor, device_ids: List[int],
                             mesh_shape: str, axis_names: str,
                             partition_spec: str):
  mesh_shape_eval = ast.literal_eval(mesh_shape)
  axis_names_eval = ast.literal_eval(axis_names)
  partition_spec_eval = ast.literal_eval(partition_spec)
  mesh = xs.Mesh(device_ids, mesh_shape_eval, axis_names_eval)
  return xs._mark_sharding(t, mesh, partition_spec_eval)


@impl(XLA_LIB, "dynamo_mark_sharding", "CompositeExplicitAutograd")
def dynamo_mark_sharding(t: torch.Tensor, device_ids: List[int],
                             mesh_shape: str, axis_names: str,
                             partition_spec: str):
  # Do nothing for non-xla tensor.
  return t
