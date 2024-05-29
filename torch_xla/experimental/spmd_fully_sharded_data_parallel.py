from typing import (Any, Callable, Dict, Optional, Union)
import warnings

import torch
import torch.nn as nn
from torch._prims_common import TensorLike, TensorSequenceType

import numpy as np

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as spmd
from torch_xla.distributed.fsdp.wrap import recursive_wrap


def _prepare_spmd_partition_spec(param,
                                 extra_data_axis=None,
                                 shard_maximal=False):
  shape = param.shape
  partition_spec = [None] * len(shape)
  # Skip scalar tensors and it replicated.
  if len(partition_spec) == 0:
    return partition_spec

  # Shard the 0th dimension of the parameter according to the
  # fsdp axis of the mesh, if shard_maximal is not specified.
  index = 0
  if shard_maximal:
    index = shape.index(max(shape))

  partition_spec[index] = "fsdp"
  if extra_data_axis:
    partition_spec[index] = (extra_data_axis, "fsdp")
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

  def __init__(
      self,
      module: nn.Module,
      *,
      mesh: Optional[spmd.Mesh] = None,
      shard_output: Optional[Callable] = None,
      auto_wrap_policy: Optional[Callable] = None,
      auto_wrapper_callable: Optional[Callable] = None,
      extra_data_axis: Optional[str] = None,
  ):
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
    if mesh is None:
      mesh = spmd.get_global_mesh()
      if mesh is None:
        raise ValueError(
            "No mesh is provided and no global mesh is set. Please provide a mesh."
        )
    if "fsdp" not in mesh.axis_names:
      raise ValueError("The mesh must have an axis named 'fsdp'.")
    if extra_data_axis and extra_data_axis not in mesh.axis_names:
      raise ValueError(
          f"The provided {extra_data_axis} axis is not in the mesh.")

    super().__init__()

    wrapper_cls = auto_wrapper_callable or SpmdFullyShardedDataParallel
    if auto_wrap_policy is not None:
      auto_wrap_kwargs = {
          "module": module,
          "auto_wrap_policy": auto_wrap_policy,
          "wrapper_cls": wrapper_cls,
          "ignored_modules": [],
          "ignored_params": [],
          "only_wrap_children": True,  # avoid double wrapping the root
      }
      fsdp_kwargs = dict(
          mesh=mesh,
          shard_output=shard_output,
          # `auto_wrap_policy` doesn't need to be specified in auto-wrapping
          # `auto_wrapper_callable`` doesn't need to be specified in auto-wrapping
      )
      self._auto_wrap(auto_wrap_kwargs, fsdp_kwargs)

    # Let's move the module to xla device in case it's not moved
    # by the caller already.
    self._orig_module = module.to(xm.xla_device())
    self._mesh = mesh

    # Only handle params which are not already sharded. This enables
    # sharding individual layers of a Module, with an outer wrapper to
    # shard any leftover parameters.
    for param in module.parameters():
      if torch_xla._XLAC._get_xla_sharding_spec(param) != "":
        continue
      spmd.mark_sharding(
          param, mesh, _prepare_spmd_partition_spec(param, shard_maximal=True))

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

        spmd.mark_sharding(
            real_output, mesh,
            _prepare_spmd_partition_spec(real_output, extra_data_axis))

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

  def _auto_wrap(
      self,
      auto_wrap_kwargs: Dict[str, Any],
      fsdp_kwargs: Dict[str, Any],
  ) -> None:
    """
    Recursively auto wraps the root module given by the key "module" in
    ``auto_wrap_kwargs`` with the arguments in ``auto_wrap_kwargs`` and
    ``fsdp_kwargs``.
    Precondition: ``auto_wrap_policy`` contains the arguments expected by
    ``_recursive_wrap()``, where ``auto_wrap_policy`` is not ``None``.
    ``fsdp_kwargs`` contains all FSDP arguments except ``module``.
    """
    auto_wrap_policy = auto_wrap_kwargs["auto_wrap_policy"]
    root_module = auto_wrap_kwargs["module"]
    assert auto_wrap_policy is not None
    # For auto wrapping, submodules should not already be wrapped with FSDP
    # since double wrapping is not supported
    for module_name, module in root_module.named_modules():
      if isinstance(module, SpmdFullyShardedDataParallel):
        raise ValueError(
            f"Expected {module_name} to NOT be SpmdFullyShardedDataParallel "
            "if using an `auto_wrap_policy`")

    recursive_wrap(**auto_wrap_kwargs, **fsdp_kwargs)
