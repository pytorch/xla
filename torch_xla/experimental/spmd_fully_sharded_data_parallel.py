from typing import (Any, Callable, Optional, Union)
import warnings

import torch
import torch.nn as nn
from torch._prims_common import TensorLike, TensorSequenceType

import numpy as np

import torch_xla
import torch_xla.distributed.spmd as spmd


def _prepare_spmd_partition_spec(param):
  partition_spec = [None] * len(param.shape)
  # Skip scalar tensors and it replicated.
  if len(partition_spec) == 0:
    return partition_spec

  # Only shard the 0th dimension of the parameter according to the
  # fsdp axis of the mesh.
  # TODO: should we shard on the maximal dim for param? Then we need
  # another helper for the output.
  partition_spec[0] = "fsdp"
  return tuple(partition_spec)


class SpmdFullyShardedDataParallel(nn.Module):
  """
  This is an experiemntal implementation of rewriting FullyShardedDataParallel using SPMD.
  The usage is similar to FSDP, but with some subtle differences args.

  Args:
    module: The module to be wrapped.
    mesh: The mesh to be used for sharding.
    shard_output: A callable to shard the output of the forward pass.
      The callable should have the signature (output, mesh) -> None.
      If None, the default implementation will shard the first tensor in the output.
      If the output is a tuple, only the first tensor will be sharded.
  """

  def __init__(self,
               module: nn.Module,
               mesh: spmd.Mesh,
               shard_output: Optional[Callable] = None):
    if isinstance(module, SpmdFullyShardedDataParallel):
      raise RuntimeError(
          "Cannot wrap a module that is already wrapped with FSDP. For nested FSDP, "
          "first wrap the inner child modules before wrapping the outer parent module."
      )
    is_forward_defined = (
        hasattr(module, "forward") and hasattr(module.forward, "__func__") and
        module.forward.__func__ != nn.Module.forward)
    if not is_forward_defined:
      raise RuntimeError(
          "The module wrapped by FSDP *must define a `forward` method and call it "
          "during the module's forward pass for FSDP to work correctly.* "
          "Hence, do not wrap `nn.ModuleList` or `nn.ModuleDict` with FSDP "
          "(since they don't have `forward` defined), "
          "and do not perform the forward pass in other ways apart from the `forward` method. "
          "(i.e. you should directly call the FSDP-wrapped module itself in your code, "
          "instead of using any of its submodules or its weights).")
    if "fsdp" not in mesh.axis_names:
      raise ValueError("The mesh must have an axis named 'fsdp'.")

    super().__init__()

    self._orig_module = module
    self._mesh = mesh

    # Only handle params which are not already sharded. This enables
    # sharding individual layers of a Module, with an outer wrapper to
    # shard any leftover parameters.
    for param in module.parameters():
      if torch_xla._XLAC._get_xla_sharding_spec(param) != "":
        continue
      spmd.mark_sharding(param, mesh, _prepare_spmd_partition_spec(param))

    # Register a backward hook to place optimization barrier to prevent
    # gigantic fusions on syncing the gradients.
    spmd.xla_sharding.apply_backward_optimization_barrier(module)

    # Need to shard the output of the forward to instruct the compiler
    # to enforce the FSDP algorithm.
    if shard_output is None:

      def shard_output_impl(output, mesh):
        real_output = None
        if isinstance(output, TensorLike):
          real_output = output
        elif isinstance(output, tuple):
          real_output = output[0] if isinstance(output[0], TensorLike) else None
          warnings.warn(
              "The output is a tuple, but only the first element is sharded. If this is not intended, please provide your own shard_output callable."
          )
        if real_output is None:
          raise RuntimeError(
              f"The output type is not supported: {type(output)}. Please provide your own shard_output callable."
          )

        spmd.mark_sharding(real_output, mesh,
                           _prepare_spmd_partition_spec(real_output))

      shard_output = shard_output_impl

    self._shard_output = shard_output

  @property
  def module(self) -> nn.Module:
    """make model.module accessible, just like DDP."""
    return self._orig_module

  def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
    output = self.module(*args, **kwargs)
    # Need to shard the output of the forward to instruct the compiler
    # to enforce the FSDP algorithm.
    self._shard_output(output, self._mesh)
    return output

  def __getattr__(self, name: str) -> Union[torch.Tensor, nn.Module]:
    """Forward missing attributes to wrapped module."""
    try:
      return super().__getattr__(name)  # defer to nn.Module's logic
    except AttributeError:
      return getattr(self.module, name)

  def __getitem__(self, key: int) -> nn.Module:
    """Forward indexing calls in case the module is a nn.Sequential."""
    return self.module.__getitem__(key)
