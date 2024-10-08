# This file is largely inspired by and mostly follows the structure of
# ``fairscale.nn.FullyShardedDataParallel`` in
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/data_parallel/fully_sharded_data_parallel.py

from collections import OrderedDict
import contextlib
from enum import Enum, auto
import functools
import gc
from itertools import chain
import logging
from math import inf
import time
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm

from .xla_flatten_params_wrapper import XlaFlattenParamsWrapper
from .utils import (
    BucketizedReduceScatter,
    DummyReduceScatter,
    dummy_all_gather,
    dummy_all_reduce,
    apply_xla_patch_to_nn_linear,
)

from .wrap import recursive_wrap
from ._init_utils import _materialize_module

import os

XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))

from torch_xla.utils.checkpoint import chkpt_status

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


class TrainingState(Enum):
  """
  Simple enum to indicate what state FSDP is in. Used for asserting
  to make sure APIs are called in the correct state.

  ..note::

      BACKWARD_PRE and BACKWARD_POST states are used to ensure we
      receives backward hooks in the correct order. It is used to catch
      unexpected order of hooks being called (likely due to our
      hook registration logic or autograd engine logic changes).
  """

  IDLE = auto()
  FORWARD = auto()
  BACKWARD_PRE = auto()
  BACKWARD_POST = auto()


class XlaFullyShardedDataParallel(nn.Module):
  """
  A wrapper for sharding Module parameters across data parallel workers in
  PyTorch XLA. XlaFullyShardedDataParallel is commonly shorten to FSDP.

  The implementation of this class is largely inspired by and mostly follows
  the structure of ``fairscale.nn.FullyShardedDataParallel`` in
  https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html.

  Pseudo-code usage::

      sharded_module = XlaFullyShardedDataParallel(my_module)
      optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
      output = sharded_module(x, y)
      loss = output.sum()
      loss.backward()
      optim.step()

  It is also possible to shard individual layers separately and have an outer
  wrapper handle any leftover parameters. This can be helpful to further
  reduce XLA device memory usage and CPU memory usage when initializing large
  models and to improve training speed by overlapping the all-gather step
  across the forward pass.

  .. warning::

      The optimizer must be initialized *after* the module has been wrapped,
      since FSDP will shard parameters in-place and this will break any
      previously initialized optimizers.

  .. warning::

      Please use ``optim.step()`` instead of ``xm.optimizer_step(optim)`` for
      optimizer update. The latter averages the gradients across XLA devices,
      which is incorrect for FSDP.

  .. warning::

      When saving checkpoints, the training process on each XLA device needs
      to save its own (sharded) model and optimizer state_dict to a different
      path. *To consolidate sharded checkpoints later, please also save
      ``model.get_shard_metadata()``* along with ``model.state_dict()`` and
      ``optimizer.state_dict()`` as follows:

          ckpt = {
              'model': model.state_dict(),
              'shard_metadata': model.get_shard_metadata(),
              'optimizer': optimizer.state_dict(),
          }
          ckpt_path = f'/tmp/rank-{xr.global_ordinal()}-of-{xr.world_size()}.pth'
          xm.save(ckpt, ckpt_path, master_only=False)

      When resuming training of an FSDP model from saved checkpoints, all
      training processes need to load their corresponding (sharded) model and
      optimizer state_dict. Use ``consolidate_sharded_model_checkpoints`` or
      run ``python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts``
      build a full model state_dict for the original unwrapped module from
      the sharded model state_dict.

  Args:
      module (nn.Module):
          module to be wrapped with FSDP. If the input module's parameters
          and buffers are not already on XLA device, they will be cast to
          ``xm.xla_device()`` (after sharding) during FSDP initialization.
      reshard_after_forward (bool, Optional):
          if ``True``, reshard parameters after the forward pass. This saves
          memory but slows training. This is only relevant when resharding
          individual layers.
      flatten_parameters (bool, Optional):
          if ``True``, flatten parameters into a single contiguous tensor for
          all_gather and reduce_scatter, which could potentially improve speed.
          In this case, one cannot apply separate optimizer groups to different
          original parameters in the wrapped module (e.g. setting bias terms or
          any BatchNorm submodules to have zero weight decay) since all the
          original parameters now become a single concatenated vector.
      execute_sharding_on_init (bool, Optional):
          if ``True``, immediately execute the parameter sharding via
          `xm.mark_step` to free up the memory of the full parameters.
      optimization_barrier_in_forward (bool, Optional):
          if ``True``, apply `xm.optimization_barrier_` on the FSDP module's
          inputs and outputs. This avoids XLA fusion with other forward pass
          computation outside the FSDP module and could save additional memory.
      optimization_barrier_in_backward (bool, Optional):
          if ``True``, apply `xm.optimization_barrier_` on the FSDP module's
          backward incoming gradients. This avoids XLA fusion with other
          backward pass computation outside the FSDP module and could save
          additional memory.
      mark_step_on_finalization (bool, Optional):
          if ``True``, call `xm.mark_step` upon finalizing gradients in the
          root FSDP module. Here in `xm.mark_step` is only called once for the
          entire backward pass and should therefore only moderately increase
          the execution time. When setting to ``True``, this option may help
          prevent undesired fusion in backward pass and save more memory.
      disable_reshard_on_root (bool, Optional):
          If ``True``, ``reshard_after_forward`` will be set to ``False`` if
          the module is a FSDP root module to improve performance. For some
          cases, we do not reshard the full parameters of an FSDP root module
          since those parameters are needed immediately for the backward pass.
          If ``False``, the performance will be lower, but it is needed because
          it helps to save memory. Consider a case that an FSDP root module is
          a submodule of a model. Backward pass may not start immediate after
          the FSDP root module finishes its forward. So, reshard the parameters
          for the FSDP root modules can help to save memory in this case.
          Default: True.
      compute_dtype (torch.dtype, Optional):
          dtype for full parameters for computation. This defaults to
          ``torch.float32`` but can be set to ``torch.float16`` or
          ``torch.bfloat16``. The sharded parameters will always be in FP32.
      buffer_dtype (torch.dtype, Optional):
          dtype for buffers for computation. This defaults to ``compute_dtype``.
      fp32_reduce_scatter (bool, Optional):
          if ``True``, then reduce-scatter gradients in FP32. This is only
          relevant when *``compute_dtype``* is not ``torch.float32``.
      sharding_groups (list, Optional):
          If specified, FSDP will use this ``sharding_groups`` for all-gather
          and reduce-scatter ops in full parameter construction and gradient
          sharding. This can be useful for mixing FSDP with model parallelism
          such as Megatron. One must also specify ``sharding_rank`` and
          ``sharding_world_size`` when using ``sharding_groups``.
      sharding_rank (int, Optional):
          The rank of this sharding instance. This must be specified if
          ``sharding_groups`` is provided. Otherwise it defaults to
          ``xr.global_ordinal()``.
      sharding_world_size (int, Optional):
          The world_size of this sharding instance. This must be specified if
          ``sharding_groups`` is provided. Otherwise it defaults to
          ``xr.world_size()``.
      pin_layout_in_collective_ops (bool, Optional):
          if ``True``, then pin the layout in the collective ops (all_reduce,
          all_gather, and reduce_scatter) in FSDP. See `xm.all_reduce` for
          details on pinning layout.
      shard_param_on_dim_0 (bool, Optional):
          if ``True``, then shard the parameter tensors only along their first
          dimension (dim 0) *without* flattening them. This is a workaround for
          those compilers that may have trouble handling flattened parameters.
          This option has no effect if ``flatten_parameters`` is ``True``.
      auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]):
          A callable specifying a policy to recursively wrap layers with FSDP.
          Note that this policy currently will only apply to child modules of
          the passed in module. The remainder modules are always wrapped in
          the returned FSDP root instance.
          ``size_based_auto_wrap_policy`` in ``torch_xla.distributed.fsdp.wrap``
          is an example of ``auto_wrap_policy`` callable, this policy wraps
          layers with the number of parameters larger than 100M.
          ``transformer_auto_wrap_policy`` in ``torch_xla.distributed.fsdp.wrap``
          is an example of ``auto_wrap_policy`` callable for transformer-like
          model architectures. Users can supply the customized
          ``auto_wrap_policy`` callable that should accept following arguments:
          ``module: nn.Module``, ``recurse: bool``, ``unwrapped_params: int``,
          and return a ``bool`` specifying whether the passed in ``module``
          should be wrapped (if ``recurse=False``) or whether we should recurse
          down the subgraph of ``module`` children (if ``recurse=True``).
          Extra customized arguments could be added to the customized
          ``auto_wrap_policy`` callable as well. It is a good practice to print
          out the sharded model and check whether the sharded model is what the
          application wants and then adjust accordingly.
          Example::

              def custom_auto_wrap_policy(
                  module: nn.Module,
                  recurse: bool,
                  unwrapped_params: int,
                  # These are customizable for this policy function.
                  min_num_params: int = int(1e8),
              ) -> bool:
                  return unwrapped_params >= min_num_params
              # Configure a custom min_num_params
              auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=1e5)

      auto_wrapper_callable (Optional[Callable]): the wrapper class or callable
          used in auto_wrap_policy (default is `XlaFullyShardedDataParallel`)
          to when wrapping a submodule. One can specify a different callable
          as wrapper. For example, activation checkpointing (rematerialization)
          can be applied to each auto-wrapped submodule as follows:

              from torch_xla.distributed.fsdp import checkpoint_module
              auto_wrapper_callable = lambda m, *args, **kwargs: XlaFullyShardedDataParallel(
                  checkpoint_module(m), *args, **kwargs)

        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should be initialized
            onto an actual device. Note that as of v1.12, we detect modules on the meta
            device via ``is_meta`` check and apply a default initialization that calls
            ``reset_parameters`` method on the passed in ``nn.Module`` if ``param_init_fn``
            is not specified, otherwise we run ``param_init_fn`` to initialize the passed
            in ``nn.Module``. In particular, this means that if ``is_meta=True`` for any
            module parameters for modules that will be wrapped with FSDP and ``param_init_fn``
            is not specified, we assume your module properly implements a ``reset_parameters()``
            and will throw errors if not. Note that additionally, we offer support for modules
            initialized with torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init`` API. In this case, deferred modules would be initialized
            by a default initialization function that calls torchdistX's
            ``materialize_module``, or the passed in ``param_init_fn``, if it is not
            ``None``. The same ``Callable`` is applied to initialize all meta modules.
            Note that this initialization function is applied before doing any FSDP sharding
            logic. And the torchdistX is an experimental package that is not fully tested in the CI.
            Example::
                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module):
                >>>     # responsible for initializing a module, such as with reset_parameters
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)
  """

  def __init__(
      self,
      module: nn.Module,
      reshard_after_forward: bool = True,
      flatten_parameters: bool = False,
      execute_sharding_on_init: bool = True,
      optimization_barrier_in_forward: bool = True,
      optimization_barrier_in_backward: bool = True,
      mark_step_on_finalization: bool = False,
      disable_reshard_on_root: bool = True,
      compute_dtype: Optional[torch.dtype] = None,
      buffer_dtype: Optional[torch.dtype] = None,
      fp32_reduce_scatter: bool = False,
      sharding_groups: Optional[List[List[int]]] = None,
      sharding_rank: Optional[int] = None,
      sharding_world_size: Optional[int] = None,
      shard_param_on_dim_0: bool = False,
      pin_layout_in_collective_ops: bool = True,
      reduce_scatter_bucket_size_mb: Optional[int] = 0,
      coalesce_all_gather_ops: bool = False,
      auto_wrap_policy: Optional[Callable] = None,
      auto_wrapper_callable: Optional[Callable] = None,
      param_init_fn: Optional[Callable[[nn.Module], None]] = None,
      _shard_size_multiple: int = 128,
      _use_xla_patched_linear: bool = True,
      _debug_dummy_forward_pass: bool = False,
      _debug_msg: str = "xla_fsdp",
      _debug_print: bool = False,
      _debug_dummy_all_gather_op: bool = False,
      _debug_dummy_all_reduce_op: bool = False,
      _debug_dummy_reduce_scatter_op: bool = False,
      _debug_dummy_optimization_barrier_op: bool = False,
  ):
    if isinstance(module, XlaFullyShardedDataParallel):
      raise RuntimeError(
          "Cannot wrap a module that is already wrapped with FSDP. For nested FSDP, "
          "first wrap the inner child modules before wrapping the outer parent module."
      )
    is_forward_defined = (
        hasattr(module, "forward") and hasattr(module.forward, "__func__") and
        module.forward.__func__ != torch.nn.Module.forward)
    if not is_forward_defined:
      raise RuntimeError(
          "The module wrapped by FSDP *must define a `forward` method and call it "
          "during the module's forward pass for FSDP to work correctly.* "
          "Hence, do not wrap `nn.ModuleList` or `nn.ModuleDict` with FSDP "
          "(since they don't have `forward` defined), "
          "and do not perform the forward pass in other ways apart from the `forward` method. "
          "(i.e. you should directly call the FSDP-wrapped module itself in your code, "
          "instead of using any of its submodules or its weights).")

    super().__init__()

    wrapper_cls = auto_wrapper_callable or XlaFullyShardedDataParallel
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
          reshard_after_forward=reshard_after_forward,
          flatten_parameters=flatten_parameters,
          execute_sharding_on_init=execute_sharding_on_init,
          optimization_barrier_in_forward=optimization_barrier_in_forward,
          optimization_barrier_in_backward=optimization_barrier_in_backward,
          mark_step_on_finalization=mark_step_on_finalization,
          disable_reshard_on_root=disable_reshard_on_root,
          compute_dtype=compute_dtype,
          buffer_dtype=buffer_dtype,
          fp32_reduce_scatter=fp32_reduce_scatter,
          sharding_groups=sharding_groups,
          sharding_rank=sharding_rank,
          sharding_world_size=sharding_world_size,
          shard_param_on_dim_0=shard_param_on_dim_0,
          pin_layout_in_collective_ops=pin_layout_in_collective_ops,
          # `auto_wrap_policy` doesn't need to be specified in auto-wrapping
          # `auto_wrapper_callable`` doesn't need to be specified in auto-wrapping
          param_init_fn=param_init_fn,
          _shard_size_multiple=_shard_size_multiple,
          _use_xla_patched_linear=_use_xla_patched_linear,
          _debug_dummy_forward_pass=_debug_dummy_forward_pass,
          _debug_msg=_debug_msg,
          _debug_print=_debug_print,
          _debug_dummy_all_gather_op=_debug_dummy_all_gather_op,
          _debug_dummy_all_reduce_op=_debug_dummy_all_reduce_op,
          _debug_dummy_reduce_scatter_op=_debug_dummy_reduce_scatter_op,
          _debug_dummy_optimization_barrier_op=_debug_dummy_optimization_barrier_op,
      )
      self._auto_wrap(auto_wrap_kwargs, fsdp_kwargs)

    self.reshard_after_forward = self._orig_reshard_after_forward = reshard_after_forward
    self.disable_reshard_on_root = disable_reshard_on_root
    self.flatten_parameters = flatten_parameters
    self.optimization_barrier_in_forward = optimization_barrier_in_forward
    self.optimization_barrier_in_backward = optimization_barrier_in_backward
    self.mark_step_on_finalization = mark_step_on_finalization

    if compute_dtype is not None and compute_dtype not in FLOAT_DTYPES:
      raise ValueError(
          f"compute_dtype must be one of {FLOAT_DTYPES}, not {compute_dtype}")
    self.compute_dtype = compute_dtype or torch.float32
    if buffer_dtype is not None and buffer_dtype not in FLOAT_DTYPES:
      raise ValueError(
          f"buffer_dtype must be one of {FLOAT_DTYPES}, not {buffer_dtype}")
    self.buffer_dtype = buffer_dtype or self.compute_dtype
    self.fp32_reduce_scatter = fp32_reduce_scatter

    # Make sharded parameter sizes a multiple of 128 for efficient all_gather ops on TPUs
    # (see https://github.com/pytorch/xla/issues/3510#issuecomment-1101739677 for details)
    self._shard_size_multiple = _shard_size_multiple if not shard_param_on_dim_0 else 1
    # Use a patched version of `torch.nn.functional.linear` with explicitly-defined backward in XLA
    # (see https://github.com/pytorch/xla/issues/3811 for details)
    self._use_xla_patched_linear = _use_xla_patched_linear
    # A workaround for those compilers that have trouble addressing flattened parameters
    # (see https://github.com/pytorch/xla/pull/3830#discussion_r939438914 for details)
    # When `_shard_param_on_dim_0` is True, we shard and all-gather model parameter tensors
    # only along their dim 0 without flattening the parameter
    self._shard_param_on_dim_0 = shard_param_on_dim_0 and not flatten_parameters
    # Allow specifying groups for the sharding collective ops, useful for mixing
    # FSDP data parallelism with model parallelism (e.g. Megatron)
    self.sharding_groups = sharding_groups
    if sharding_groups is None:
      self.rank = xr.global_ordinal()
      self.world_size = xr.world_size()
    else:
      if sharding_rank is None or sharding_world_size is None:
        raise ValueError(
            "sharding_rank and sharding_world_size must be provided when sharding_groups is specified"
        )
      self.rank = sharding_rank
      self.world_size = sharding_world_size

    self.coalesce_all_gather_ops = coalesce_all_gather_ops
    # Set layout pinning to False in all_gather, all_reduce, and reduce_scatter so that they can work together
    # TODO (ronghanghu): change the default layout pinning to True after it's supported simultaneously
    # on all collective ops (see https://github.com/pytorch/xla/pull/3511 for details)
    if _debug_dummy_all_gather_op:
      self.all_gather_op = dummy_all_gather
    else:
      self.all_gather_op = functools.partial(
          xm.all_gather, pin_layout=pin_layout_in_collective_ops)
    if _debug_dummy_all_reduce_op:
      self.all_reduce_op = dummy_all_reduce
    else:
      self.all_reduce_op = functools.partial(
          xm.all_reduce, pin_layout=pin_layout_in_collective_ops)
    if _debug_dummy_reduce_scatter_op:
      self.reduce_scatter_op = DummyReduceScatter(shard_count=self.world_size)
    else:
      self.reduce_scatter_op = BucketizedReduceScatter(
          reduce_scatter_bucket_size_mb,
          shard_count=self.world_size,
          groups=self.sharding_groups,
          pin_layout=pin_layout_in_collective_ops)
    if _debug_dummy_optimization_barrier_op:
      self.optimization_barrier_op = lambda *args: None
    else:
      self.optimization_barrier_op = xm.optimization_barrier_

    # Allow specifying groups for the sharding collective ops, useful for mixing
    # FSDP data parallelism with model parallelism (e.g. Megatron)
    self.sharding_groups = sharding_groups
    if sharding_groups is None:
      self.rank = xr.global_ordinal()
      self.world_size = xr.world_size()
    else:
      if sharding_rank is None or sharding_world_size is None:
        raise ValueError(
            "sharding_rank and sharding_world_size must be provided when sharding_groups is specified"
        )
      self.rank = sharding_rank
      self.world_size = sharding_world_size

    # Options for debugging
    # - set _debug_dummy_forward_pass=True to check for parameter-only memory consumption
    # - set _debug_msg="xxx" and _debug_print=True to distinguish different FSDP instance
    self._debug_dummy_forward_pass = _debug_dummy_forward_pass
    self._debug_msg = _debug_msg
    self._debug_print = _debug_print

    self.gradient_predivide_factor: float = self._get_gradient_predivide_factor(
        self.world_size)
    self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

    self._tstart = time.time()

    if self._use_xla_patched_linear:
      # Use a patch to `nn.Linear` (`torch.nn.functional.linear`) in XLA so that its
      # backward pass will use its weight parameter rather than an intermediate result.
      # (see https://github.com/pytorch/xla/issues/3811 for details)
      module = apply_xla_patch_to_nn_linear(module)

    _materialize_module(
        module,
        param_init_fn,
        [],  # TODO: ignored_params is set to empty now, pass in correct params when this feature is fully enabled
        deferred_init_check_fn=lambda k: not isinstance(k, wrapper_cls))

    # Only handle params which are not already sharded. This enables
    # sharding individual layers of a Module, with an outer wrapper to
    # shard any leftover parameters.
    params = []
    for param in module.parameters():
      if not hasattr(param, "_is_sharded"):
        params.append(param)

    # For now, it is either all flatten or none flatten.
    if self.flatten_parameters:
      # separately flatten trainable and frozen parameters
      trainable_params = [p for p in params if p.requires_grad]
      frozen_params = [p for p in params if not p.requires_grad]
      to_be_flatten_params: List[List[Parameter]] = [trainable_params]
      if len(frozen_params) > 0:
        to_be_flatten_params.append(frozen_params)
      non_flatten_params = []
    else:
      to_be_flatten_params: List[List[Parameter]] = [[]]
      non_flatten_params = params

    # Here, we don't automatically unflatten XlaFlattenParamsWrapper's state dict
    # to avoid overhead on XLA devices. Use ``get_shard_metadata`` to save parameter info
    # ``consolidate_sharded_model_checkpoints`` to consolidate the sharded checkpoints.
    self._fsdp_wrapped_module: nn.Module = XlaFlattenParamsWrapper(
        module,
        param_list=to_be_flatten_params,
        auto_unflatten_state_dict=False,
    )
    del module  # free original module in case it helps garbage collection

    # Now, in this FSDP wrapper class, we keep a list of to-be-flatten and not-to-be-flatten
    # params for doing sharding, gradient hooks, etc. Note, the ordering of the
    # list matters: flatten params are always in the front.
    params_to_shard = cast(
        List[Parameter],
        self._fsdp_wrapped_module.flat_params) + non_flatten_params

    self.xla_device = xm.xla_device()
    # Shard module parameters in place
    self._shard_parameters_(params_to_shard)
    # Cast the module buffers to the specified buffer_dtype
    self._cast_buffers(self.buffer_dtype)

    # Make sure all parameters are sharded.
    for n, p in self.named_parameters():
      assert hasattr(
          p, "_is_sharded"), f"found unsharded parameter: {n} ; {p.size()}"

    self._reset_lazy_init()

    # Flag to indicate if we require gradient reduction in the backward
    # pass. This will be False when inside the no_sync context manager.
    self._require_backward_grad_sync: bool = True

    # Enum to indicate if we're in the forward/backward pass, idle, etc.
    self.training_state = TrainingState.IDLE

    # Flag to indicate if the full params are gathered.
    self.has_full_params: bool = False

    # Flag to guard against preparing gradients multiple times per iteration.
    # This is reset at the end of the backward pass.
    self._pre_backward_hook_has_run = False

    if execute_sharding_on_init:
      # Execute the parameter sharding immediately and free up the memory
      gc.collect()
      xm.mark_step()

  def _get_gradient_predivide_factor(self, world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
      factor *= 2
    return float(factor)

  def set_gradient_divide_factors(self, pre: float, post: float,
                                  recursive: bool) -> None:
    """
    Allowing user to override the pre and post divide factors.

    Args:
        pre (float): divide factor before the reduction.
        post (float): divide factor after the reduction.
        recursive (bool): recursively set it for all child FSDP instances or not.
    """
    self.assert_state(TrainingState.IDLE)
    if recursive:
      for module in self.modules():
        if isinstance(module, XlaFullyShardedDataParallel) and module != self:
          module.set_gradient_divide_factors(pre, post, False)
    self.gradient_predivide_factor = pre
    self.gradient_postdivide_factor = post
    if (pre, post) == (1, 1):
      self.reduce_scatter_op.scale = 1.0 / self.world_size
    else:
      self.reduce_scatter_op.scale = 1.0

  @property
  def module(self) -> XlaFlattenParamsWrapper:
    """make model.module accessible, just like DDP."""
    assert isinstance(self._fsdp_wrapped_module, XlaFlattenParamsWrapper)
    return self._fsdp_wrapped_module

  @property
  def params_with_grad(self) -> List[Parameter]:
    """[p for p in self.parameters() if p.grad is not None]"""
    return [p for p in self.parameters() if p.grad is not None]

  @torch.no_grad()
  def clip_grad_norm_(
      self,
      max_norm: Union[float, int],
      norm_type: Union[float, int] = 2.0,
      groups: Optional[List[List[int]]] = None,
  ) -> torch.Tensor:
    """
    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.

    Args:
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
            for infinity norm.
        groups (list, optional): A list of list, representing the replica
            groups for the all-reduce operation to compute global norms.
            See `xm.all_reduce` for details.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    .. note:: This is analogous to `torch.nn.utils.clip_grad_norm_` but
        handles the partitioning and multiple devices per rank under the
        hood. The default torch util is not applicable here, because each
        rank only has a partial view of all the grads in the model, so
        calling it in the OSS context would lead to different scaling being
        applied per subset of model parameters.

    .. warning:: This needs to be called on all ranks, since synchronization
        primitives will be used.
    """
    assert self._is_root, "clip_grad_norm should only be called on the root (parent) instance"
    self.assert_state(TrainingState.IDLE)

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    params_with_grad = self.params_with_grad
    # Computes the max norm for this shard's gradients and sync's across workers
    local_norm = _calc_grad_norm(params_with_grad, norm_type)
    if norm_type == inf:
      total_norm = self.all_reduce_op(xm.REDUCE_MAX, local_norm, groups=groups)
    else:
      total_norm = self.all_reduce_op(
          xm.REDUCE_SUM, local_norm**norm_type, groups=groups)
      total_norm = total_norm**(1.0 / norm_type)

    # Now multiply each grad by (max_norm/total_norm), same as torch 1.7 https://tinyurl.com/3wtxhhqq)
    clip_coef = torch.clip(max_norm / (total_norm + 1e-6), 0.0, 1.0)
    for p in params_with_grad:
      p.grad.detach().mul_(clip_coef)

    return total_norm

  @torch.no_grad()
  def _shard_parameters_(self, params_to_shard) -> None:
    """
    At initialization we wrap a module with full parameters and shard the
    parameters in-place. Sharding is implemented by viewing each parameter
    as a 1D Tensor and retaining only a single slice, where the slice size
    is determined by the number of data parallel workers.

    Wrapping modules with many small parameters (or with a very large data
    parallel world size) will result in many small parameter shards and slow
    performance. In this case it's better to set *``flatten_parameters``* to
    ``True``, so that all of the small parameters in the module are combined
    into a single contiguous Tensor and sharded once.

    After this initial sharding is complete, the user can initialize a
    ``torch.optim.Optimizer`` in the usual way, i.e.::

    .. code-block:: python

        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

    The optimizer will see only a single slice of parameters and will thus
    allocate less memory for optimizer state, avoiding redundancy across
    data parallel workers.

    Note: this method is implemented in a different manner from
    ``fairscale.nn.FullyShardedDataParallel``. Here we delete the original
    module parameters and create new sharded parameter tensors (instead of
    making sharded tensors an attribute of the original parameters). This
    make it easier to handle things (e.g. freeing parameters) on XLA.
    """
    if len(params_to_shard) > 0:
      # When freeing the full parameters, we point their internal XLATensor to this placeholder
      # (so that the XLA compiler can reuse the memory storage).
      self._dummy_data_placeholder = torch.zeros(
          1, dtype=self.compute_dtype, device=self.xla_device)

    # get the module names of each full parameter to shard
    params_to_shard_set = set(params_to_shard)
    assert len(params_to_shard_set) == len(params_to_shard), \
        "params_to_shard should not have dups"
    full_param_infos = []
    shared_full_param_memo = {}
    shared_full_param_infos = []
    full_params = []
    for module_name, m in self.named_modules():
      for n, p in m.named_parameters(recurse=False):
        if p.dtype != torch.float32:
          raise TypeError("only fp32 parameters are supported")
        if p in params_to_shard_set:
          if p in shared_full_param_memo:
            mname, shared_m, shared_n = shared_full_param_memo[p]
            shared_full_param_infos.append(
                (module_name, mname, m, n, shared_m, shared_n))
          else:
            shared_full_param_memo[p] = (module_name, m, n)
            full_param_infos.append((module_name, m, n))
            full_params.append(p)
    assert len(full_params) == len(params_to_shard_set), \
        f"there are parameters in params_to_shard not belonging to this module."
    del shared_full_param_memo
    self.full_params = full_params
    self.full_param_infos = full_param_infos
    self.shared_full_param_infos = shared_full_param_infos

    # allocate and register new sharded parameters
    self.sharded_params = []
    for idx, (module_name, m, n) in enumerate(self.full_param_infos):
      p = self.full_params[idx]
      assert not hasattr(p, "_is_sharded")

      shard_data = self._get_shard(p)
      if shard_data.device != self.xla_device:
        # cast to XLA device if not already on XLA
        shard_data = shard_data.to(self.xla_device)
      p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
      p_shard._is_sharded = True
      p_shard._orig_size = p.size()
      p_shard._orig_name = f"{module_name}.{n}"
      p_shard._name = f"_fsdp_shard.{p_shard._orig_name}".replace(
          ".", "_FSDP_SHARD_SEPARATOR_")
      self.register_parameter(p_shard._name, p_shard)
      self.sharded_params.append(p_shard)
      if p.device != self.xla_device:
        # cast to XLA device if not already on XLA
        p = p.to(self.xla_device).requires_grad_(p.requires_grad)
        # update p in full_params since id(p) changed after the casting
        self.full_params[idx] = p
      # Free the full parameter storage (here we free its internal XLATensor) but keep the tensor itself
      # for auto-grad tracing (like `torch.autograd.Variable` before the tensor-variable merge).
      if XLA_DISABLE_FUNCTIONALIZATION:
        p.data = p.new_zeros(1)  # Old behavior before Functionalization.
      else:
        torch_xla._XLAC._replace_xla_tensor(p, p.new_zeros(1))
      p._sharded_param = p_shard  # add a handle to the sharded parameter
      p._has_full_param = False
      # deregister the full parameter tensors from their modules (so that they won't
      # appear in the FSDP model's `parameters()` or `named_parameters()` outputs;
      # only the sharded parameters should appear in the FSDP model's `parameters()`)
      assert n in m._parameters
      m._parameters.pop(n)
      object.__setattr__(m, n, p)

    # also deregister the shared parameters
    for _, _, m, n, shared_m, shared_n in self.shared_full_param_infos:
      assert n in m._parameters
      m._parameters.pop(n)
      shared_p = getattr(shared_m, shared_n)
      object.__setattr__(m, n, shared_p)

    assert len(self.sharded_params) == len(self.full_params)

  def _get_shard(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return the local shard of a full tensor."""
    tensor = self._flatten_and_pad_to_world_size(
        tensor, self.world_size * self._shard_size_multiple)
    local_size = tensor.size(0) // self.world_size
    begin, end = self.rank * local_size, (self.rank + 1) * local_size
    tensor = tensor[begin:end].clone()
    return tensor

  @torch.no_grad()
  def _cast_buffers(self,
                    dtype: Optional[torch.dtype] = None,
                    memo: Optional[Set] = None) -> None:
    """Move all buffers to the given *dtype*.

    If *dtype* is not given, then it will default to ``self.buffer_dtype``.
    In the case of nested FSDP instances, we will respect the child instance's
    ``buffer_dtype`` configuration.

    Args:
        dtype (torch.dtype, Optional):
            dtype to cast buffers to (defaults to buffer_dtype)
        memo (Set, Optional):
            set of modules that have already been processed
    """
    if memo is None:
      memo = set()
    for module in self.modules():
      if module is not self and isinstance(module, XlaFullyShardedDataParallel):
        # Allow any child FSDP instances to handle their own buffers.
        module._cast_buffers(dtype=dtype, memo=memo)
      elif module not in memo:
        memo.add(module)
        for name, buf in module.named_buffers(recurse=False):
          if buf is None:
            continue
          if torch.is_floating_point(buf):
            orig_dtype = buf.dtype
            cast_dtype = dtype or self.buffer_dtype
            if orig_dtype != cast_dtype:
              buf = buf.to(cast_dtype)
              buf._orig_dtype = orig_dtype
          if buf.device != self.xla_device:
            buf = buf.to(self.xla_device)
          setattr(module, name, buf)

  def extra_repr(self) -> str:
    repr = (f"world_size={self.world_size}, "
            f"rank={self.rank}, "
            f"compute_dtype={self.compute_dtype}, "
            f"buffer_dtype={self.buffer_dtype}, "
            f"flatten_parameters={self.flatten_parameters}, "
            f"reshard_after_forward={self.reshard_after_forward}, "
            f"sharding_groups={self.sharding_groups}")
    return repr

  def __getattr__(self, name: str) -> Union[torch.Tensor, nn.Module]:
    """Forward missing attributes to wrapped module."""
    try:
      return super().__getattr__(name)  # defer to nn.Module's logic
    except AttributeError:
      return getattr(self.module, name)

  def __getitem__(self, key: int) -> nn.Module:
    """Forward indexing calls in case the module is a nn.Sequential."""
    return self.module.__getitem__(key)

  @contextlib.contextmanager
  def no_sync(self) -> Generator:
    """
    A context manager to disable gradient synchronizations across FSDP
    processes. Within this context, gradients will be accumulated on module
    variables, which will later be synchronized in the first
    forward-backward pass after exiting the context.

    .. note:: This likely results in higher memory usage because FSDP will
        accumulate the full model gradients (instead of gradient shards)
        until the eventual sync.

    .. note:: Gradient accumulation can be done without this context,
        avoiding the extra XLA device memory overhead, but with the extra
        networking overhead.
    """
    self._lazy_init()
    assert self._is_root, "no_sync on inner FSDP is not supported"
    self.assert_state(TrainingState.IDLE)
    # This instance may wrap other FSDP instances and we
    # need to set all of them to accumulate gradients.
    old_flags = []
    for m in self.modules():  # includes self
      if isinstance(m, XlaFullyShardedDataParallel):
        old_flags.append((m, m._require_backward_grad_sync))
        m._require_backward_grad_sync = False
    try:
      yield
    finally:
      for m, old_flag in old_flags:
        assert m._require_backward_grad_sync is False
        m._require_backward_grad_sync = old_flag

  def _reset_lazy_init(self) -> None:
    """Reset instance so :func:`_lazy_init` will run on the next forward."""
    self._is_root: Optional[bool] = None
    self._all_sharded_params: Optional[Parameter] = None
    self._output_pre_backward_hook_registered: Optional[Set] = None
    self._backward_opt_barrier_tensors: Optional[List] = None
    self._backward_opt_barrier_tensor_ids: Optional[Set] = None
    self.reshard_after_forward = self._orig_reshard_after_forward

  def _lazy_init(self) -> None:
    """
    Initialization steps that should happen lazily, typically right
    before the first forward pass.
    """
    # Initialize _is_root and setup streams. These steps would ideally
    # happen in __init__, but _is_root can only be determined after the
    # entire model hierarchy is setup, thus we run it lazily.
    if self._is_root is None:
      self._set_is_root()
      self._setup_output_hook_and_backward_opt_barrier_lists()

    if self._is_root and self.disable_reshard_on_root:
      # Don't free the full params for the outer-most (root) instance,
      # since those params will be needed immediately after for the
      # backward pass.
      self.reshard_after_forward = False

  def _set_is_root(self) -> None:
    """
    If ``True``, implies that no other :class:`XlaFullyShardedDataParallel`
    instance wraps this one. Called once by :func:`_lazy_init`.
    Also sets self.children_share_process_group = True if all child
    instances share the same process group. If some child instances use a
    different process group, self.clip_grad_norm_ will raise an error.
    """
    if self._is_root is not None:
      return
    # No FSDP instance wraps this, else _is_root would be set to False.
    self._is_root = True
    self._all_sharded_params = list(self.parameters())
    if self._debug_print:
      xm.master_print(
          f"root FSDP got {len(self._all_sharded_params)} total params (_debug_msg: {self._debug_msg}).",
          flush=True)
    # If final backward callback is never been queued, state should be IDLE.
    # If final backward callback is queued, the callback should be finished
    # and the state was reset to be IDLE.
    # This should be asserted at the beginning of forward pass in the root instance only.
    # For children instances, if they are checkpointed, state will not be reset to
    # IDLE after each inner forward/backward.
    self.assert_state(TrainingState.IDLE)
    # As the root, we now set all children instances to False and
    # give them a closure to try to queue a wait_for_post_backward.
    for n, m in self.named_modules():
      # `n != ""` excludes self.
      if n != "" and isinstance(m, XlaFullyShardedDataParallel):
        # We relax the assert for non-root instance, when the nested inialized module is wrapped
        # again in FSDP later, for example after training to run inference.
        assert m._is_root is None or not m._is_root
        if m._is_root is None:
          m._is_root = False

  def _setup_output_hook_and_backward_opt_barrier_lists(self) -> None:
    """
    Set up a list to avoid registering pre-backward hooks incorrectly.
    And a list to apply optimization barrier on backward pass tensors.
    """
    assert self._is_root, "This should only be called on the root"
    self._output_pre_backward_hook_registered = set()
    self._backward_opt_barrier_tensors = []
    self._backward_opt_barrier_tensor_ids = set()
    for n, m in self.named_modules():
      if n != "" and isinstance(m, XlaFullyShardedDataParallel):
        m._output_pre_backward_hook_registered = self._output_pre_backward_hook_registered
        m._backward_opt_barrier_tensors = self._backward_opt_barrier_tensors
        m._backward_opt_barrier_tensor_ids = self._backward_opt_barrier_tensor_ids

  def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._lazy_init()

    # Start of a forward pass.
    self.training_state = TrainingState.FORWARD

    if self.compute_dtype != torch.float32:
      # Cast the input float tensors to the specified compute_dtype
      args, kwargs = _cast_floats_tensors(self.compute_dtype, *args, **kwargs)

    # All-gather full parameters.
    input_opt_barrier_tensors = []
    if self.optimization_barrier_in_forward:
      # Ensure that previous ops to build this module's inputs (which are
      # usually performed in previous modules) are finished before rebuilding
      # the full params of this FSDP module.
      input_opt_barrier_tensors = collect_tensors((args, kwargs))
    self._rebuild_full_params(
        dependency_tensors=input_opt_barrier_tensors,
        apply_opt_barrier=self.optimization_barrier_in_forward)

    # Register backward hooks to reshard params and reduce-scatter grads.
    # These need to be re-registered every forward pass.
    self._register_post_backward_hooks()

    if not self._debug_dummy_forward_pass:
      outputs = self.module(*args, **kwargs)
    else:
      # Run a dummy forward pass by summing the inputs and full parameter.
      # This can be used to debug FSDP parameter memory consumption.
      outputs = self._dummy_forward(*args, **kwargs)

    # Allgather reduction optimization: if this forward is a recompute forward
    # in checkpoint, then we do not reshard here, so that the following backward
    # does not need to do the allgather
    if self.reshard_after_forward and not chkpt_status.in_chkpt_bwd:
      output_opt_barrier_tensors = []
      if self.optimization_barrier_in_forward:
        # Ensure that the full parameters of this FSDP module are freed
        # before any new ops based on this module's outputs (which are usually
        # performed in subsequent modules) can happen.
        output_opt_barrier_tensors = collect_tensors(outputs)
      self._free_full_params(
          dependency_tensors=output_opt_barrier_tensors,
          apply_opt_barrier=self.optimization_barrier_in_forward)

    # Register pre-backward hooks to all-gather the params for the backward
    # pass (if output's grad was needed). This won't register anything if
    # we are in eval mode.
    # Some model does forward pass multiple times, we need to register the
    # pre-backward hook on every output since the last output's hook has to
    # fire first to setup for backward. However, we use ``self._pre_backward_hook_has_run``
    # to prevent repeated overhead from multiple hook callbacks.
    outputs = self._register_pre_backward_hooks(outputs)

    if self.optimization_barrier_in_backward:
      # Apply XLA compiler optimization barrier to FSDP outputs and their gradients to avoid
      # fusion across FSDP modules (which sometimes results in higher memory consumption).
      input_grad_opt_barrier_tensors = input_opt_barrier_tensors or collect_tensors(
          (args, kwargs))
      self._register_grad_opt_barrier_hooks(input_grad_opt_barrier_tensors)

    # Done with a forward pass.
    self.training_state = TrainingState.IDLE

    return outputs

  def _dummy_forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    A dummy forward pass with minimal computation that sums all inputs and
    full parameters, e.g. to debug parameter memory consumption.
    """
    outputs = torch.zeros(1, device=xm.xla_device())
    for t in chain(args, kwargs.values(), self.full_params):
      if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
        outputs = outputs + t.mean()

    # recursively run dummy forward pass on inner FSDP modules (if any)
    resursive = kwargs.pop("_xla_fsdp_dummy_forward_resursive", True)
    if resursive:
      assert self._is_root
      for m in self.modules():
        if isinstance(m, XlaFullyShardedDataParallel) and m != self:
          _m_orig_debug_dummy_forward_pass = m._debug_dummy_forward_pass
          m._debug_dummy_forward_pass = True
          outputs = m(outputs, _xla_fsdp_dummy_forward_resursive=False)
          m._debug_dummy_forward_pass = _m_orig_debug_dummy_forward_pass

    return outputs

  def _try_adding_to_backward_opt_barrier_lists(self,
                                                tensor: torch.Tensor) -> None:
    """
    Add tensor to backward pass optimization barrier list if it is not there.
    """
    if id(tensor) not in self._backward_opt_barrier_tensor_ids:
      self._backward_opt_barrier_tensor_ids.add(id(tensor))
      self._backward_opt_barrier_tensors.append(tensor)

  def _clear_backward_opt_barrier_lists(self) -> None:
    """Reset the backward pass optimization barrier list"""
    self._backward_opt_barrier_tensors.clear()
    self._backward_opt_barrier_tensor_ids.clear()

  def _register_grad_opt_barrier_hooks(
      self, dependency_tensors: List[torch.Tensor]) -> None:
    """
    Register hook to `dependency_tensors` to put their gradient tensors into
    self._backward_opt_barrier_tensors for backward pass optimization barrer.
    """
    if not torch.is_grad_enabled():
      return  # don't register hooks if grad isn't enabled

    def _grad_opt_barrier_hook(t_grad: torch.Tensor):
      self._try_adding_to_backward_opt_barrier_lists(t_grad)
      self.optimization_barrier_op([t_grad])
      return t_grad.view(t_grad.size())  # a view with barrier applied

    for t in dependency_tensors:
      if t.requires_grad:
        t.register_hook(_grad_opt_barrier_hook)

  def _register_pre_backward_hooks(self, outputs: Any) -> Any:
    """
    Register pre-backward hook to run before the wrapped module's
    backward. Hooks should be attached to all outputs from the forward.

    Returns:
        outputs: new outputs with hooks registered if they requires gradient.
    """
    if not torch.is_grad_enabled():
      return outputs  # don't register hooks if grad isn't enabled

    if self._is_root:
      # This actually means that only root instance has
      # _post_backward_callback_queued defined. Accidentally accessing this field
      # will assert on all other instances, giving us a nice bug checker.
      self._post_backward_callback_queued = False

    def _pre_backward_hook(t_grad: torch.Tensor) -> None:
      # try to queue final backward callback only once for root, so
      # that final backward callback is attached to the outer most
      # backward graph task and called after all the backward
      # calls are completed.
      if self._is_root:
        self._queue_wait_for_post_backward()

      if self.optimization_barrier_in_backward:
        self._try_adding_to_backward_opt_barrier_lists(t_grad)
      # All-gather full parameters or switching to the full params.
      # Note, ``self._rebuild_full_params`` is idempotent. So in case it is called
      # unnecessarily, it doesn't incur much overhead.
      # Allgather reduction optimization: if this backward is in checkpoint, then we
      # do not allgather here, since the previous recompute forward does not reshard
      if self.reshard_after_forward and not chkpt_status.in_chkpt_bwd:
        dependency_tensors = []
        if self.optimization_barrier_in_backward:
          # Ensure that backward pass ops of feature gradients, parameter
          # gradient and sharding, and full-param freeing (which are usually
          # performed in previous modules and are registered to
          # self._backward_opt_barrier_tensors in _grad_opt_barrier_hook,
          # _pre_backward_hook, and _post_backward_hook) are finished before
          # rebuilding the full params of this FSDP module.
          dependency_tensors = self._backward_opt_barrier_tensors
        self._rebuild_full_params(
            dependency_tensors=dependency_tensors,
            apply_opt_barrier=self.optimization_barrier_in_backward)
        self._clear_backward_opt_barrier_lists()

      # Only run the following once per iteration (i.e. in case
      # it is multiple outputs or multiple forward passes).
      if not self._pre_backward_hook_has_run:
        self._pre_backward_hook_has_run = True
        # Start of a backward pass for the first time in an iteration.
        self.assert_state([TrainingState.IDLE, TrainingState.BACKWARD_PRE])
        # Check p.grad to make sure that it is in the right shape, device, etc.
        for p, p_shard in zip(self.full_params, self.sharded_params):
          if p.grad is not None:
            assert p.grad.device == p_shard.device
            assert p.grad.size() == p_shard._orig_size

      # Transition to BACKWARD_PRE state if currently IDLE. We can transition from BACKWARD_POST
      # to IDLE when FSDP is within activation checkpointing and called multiple times, due to the
      # extra forward pass for re-computation.
      if self.training_state == TrainingState.IDLE:
        self.training_state = TrainingState.BACKWARD_PRE
      self.assert_state(
          [TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])

      if self.optimization_barrier_in_backward:
        self._try_adding_to_backward_opt_barrier_lists(t_grad)
        self.optimization_barrier_op([t_grad])
        t_grad = t_grad.view(t_grad.size())  # a view with barrier applied
      return t_grad

    _registered = 0

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
      # We don't register the pre_backward hook on the same tensor that has been
      # returned from an inner FSDP, unless it is the first one.
      nonlocal _registered
      assert self._output_pre_backward_hook_registered is not None
      if t.requires_grad and (_registered == 0 or id(t)
                              not in self._output_pre_backward_hook_registered):
        t.register_hook(_pre_backward_hook)
        self._output_pre_backward_hook_registered.add(id(t))
        _registered += 1
      return t

    # Attach hooks to Tensor outputs.
    outputs = apply_to_tensors(_register_hook, outputs)

    return outputs

  def _register_post_backward_hooks(self) -> None:
    """
    Register backward hooks to reshard params and reduce-scatter grads.

    This is called during forward pass. The goal is to attach a hook
    on each of the parameter's gradient generating function (``grad_acc``
    below) so that the hook is called *after* all gradients for that
    param are computed.

    Goals:

    1. We want the hook to fire once and only once *after* all gradients
    are accumulated for a param.
    2. If it fires more than once, we end up incorrectly shard the grad
    multiple times. (could lead to dimension too small)
    3. If it fires once but too early or doesn't fire, we leave gradients
    unsharded. (could lead to dimension too large)

    Empirically, keep the first hook register per forward pass seems to
    work the best. We do need to remove the hook at the end of the
    backward pass. Otherwise, the next forward pass will not register
    a new hook, which is needed for a new forward pass.
    """
    if not torch.is_grad_enabled():
      return  # don't register grad hooks if grad isn't enabled
    self._post_backward_hooks_to_call = 0
    for p in self.full_params:
      if p.requires_grad:
        if hasattr(p, "_shard_bwd_hook"):
          continue
        # Register a hook on the first call, empirically, autograd
        # fires it at the end for this param, which makes sense.
        p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
        assert p_tmp.grad_fn is not None
        grad_acc = p_tmp.grad_fn.next_functions[0][
            0]  # Gets its GradAccumulation object.
        handle = grad_acc.register_hook(
            functools.partial(self._post_backward_hook, p))
        p._shard_bwd_hook = (grad_acc, handle)
        self._post_backward_hooks_to_call += 1

  @torch.no_grad()
  def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
    """
    At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
    full gradient for the local batch. The reduce-scatter op will replace
    ``param.grad`` with a single shard of the summed gradient across all
    XLA devices. This shard will align with the current rank. For example::

        before reduce_scatter:
            param.grad (rank #0): [1, 2, 3, 4]
            param.grad (rank #1): [5, 6, 7, 8]

        after reduce_scatter:
            param.grad (rank #0): [6, 8]    # 1+5, 2+6
            param.grad (rank #1): [10, 12]  # 3+7, 4+8

    The local XLA device's ``optim.step`` is responsible for updating a single
    shard of params, also corresponding to the current XLA device's rank. This
    alignment is created by :func:`_shard_parameters_`, which ensures that
    the local optimizer only sees the relevant parameter shard.
    """
    # First hook callback will see PRE state. If we have multiple params,
    # then subsequent hook callbacks will see POST state.
    self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
    self.training_state = TrainingState.BACKWARD_POST
    self._post_backward_hooks_to_call -= 1
    if param.grad is None:
      if self._post_backward_hooks_to_call == 0:
        self.reduce_scatter_op.flush()
      return

    assert param.grad is not None, param.shape
    if param.grad.requires_grad:
      raise RuntimeError(
          "FSDP only works with gradients that don't require gradients")

    grad = param.grad
    if self._require_backward_grad_sync or self.reshard_after_forward:
      # Free full params. As a special case, we don't free the full params
      # when in a ``no_sync`` context (as inversely indicated by
      # ``self._require_backward_grad_sync``), since the params will not
      # get updated before the next forward. This saves networking
      # bandwidth but uses more XLA device memory.
      self._free_full_params(
          [param],
          dependency_tensors=[grad],
          apply_opt_barrier=self.optimization_barrier_in_backward)

    if not self._require_backward_grad_sync:
      if self._post_backward_hooks_to_call == 0:
        self.reduce_scatter_op.flush()
      return

    if self.gradient_predivide_factor > 1:
      # Average grad by world_size for consistency with PyTorch DDP.
      grad.div_(self.gradient_predivide_factor)

    # Shard the gradients with `reduce_scatter`.
    # Clear grad on the tensor, so any repeated gradient computations do not interfere with this reduction.
    param.grad = None
    grad_flat = self._flatten_and_pad_to_world_size(
        grad, self.world_size * self._shard_size_multiple)
    if self.optimization_barrier_in_backward:
      self.optimization_barrier_op([grad_flat])
    if grad_flat.dtype != torch.float32 and self.fp32_reduce_scatter:
      grad_flat = grad_flat.to(torch.float32)

    def reduce_scatter_done(reduced_grad):
      if reduced_grad.dtype != torch.float32:
        reduced_grad = reduced_grad.to(torch.float32)
      if self.optimization_barrier_in_backward:
        self.optimization_barrier_op([reduced_grad])
      if self.gradient_postdivide_factor > 1:
        # Average grad by world_size for consistency with PyTorch DDP.
        reduced_grad.div_(self.gradient_postdivide_factor)

      grad._has_full_param = True
      grad_flat._has_full_param = True
      self._free_full_params(
          [grad, grad_flat],
          dependency_tensors=[reduced_grad],
          apply_opt_barrier=self.optimization_barrier_in_backward)
      self._try_adding_to_backward_opt_barrier_lists(reduced_grad)

      # Accumulate into the gradient shard.
      assert hasattr(param, "_sharded_param")
      p_shard = param._sharded_param
      if p_shard.grad is None:
        p_shard.grad = reduced_grad
      else:
        assert p_shard.grad.shape == reduced_grad.shape
        assert p_shard.grad.device == reduced_grad.device
        p_shard.grad += reduced_grad

    self.reduce_scatter_op(grad_flat.detach(), reduce_scatter_done)
    if self._post_backward_hooks_to_call == 0:
      self.reduce_scatter_op.flush()

  def _queue_wait_for_post_backward(self) -> None:
    """
    Try to queue a `wait_for_post_backward` callback.

    Only called on root and only queue one callback at the beginning of
    outer most backward.
    """
    assert self._is_root
    if not self._post_backward_callback_queued:
      self.assert_state([TrainingState.IDLE])
      self._post_backward_callback_queued = True
      Variable._execution_engine.queue_callback(
          self._try_wait_for_post_backward)

  def _try_wait_for_post_backward(self) -> None:
    """
    Catch and print any exception in `_wait_for_post_backward`. Otherwise the
    exception is not printed and error is very confusing as shown below.
    ```
    built-in method run_backward of torch._C._EngineBase object at 0x7f26dc335aa0> returned NULL without setting an error
    ```
    """
    try:
      self._wait_for_post_backward()
    except Exception as e:
      print(
          f"Exception below occurred in post-backward (_debug_msg: {self._debug_msg}). "
          f"This is often due to some inner FSDP modules not being used "
          f"in an outer FSDP module's forward pass. Please make sure that all inner "
          f"FSDP modules participate in the forward pass when using nested FSDP.\n"
          f"{type(e).__name__}: {e}")
      raise

  @torch.no_grad()
  def _wait_for_post_backward(self) -> None:
    """Wait for post-backward to finish. Only called on root instance."""
    assert self._is_root
    # Check if the root module has params and if any of them has
    # the `requires_grad` field set. If `requires_grad=False` for
    # all the params, the post_backward hook will not fire and the
    # state will remain in `TrainingState.BACKWARD_PRE`.
    if any([p.requires_grad for p in self.full_params]):
      self.assert_state(TrainingState.BACKWARD_POST)
    else:
      self.assert_state(TrainingState.BACKWARD_PRE)

    # A backward pass is done, clean up below.
    def _finalize_parameters(fsdp_module: XlaFullyShardedDataParallel) -> None:
      """Helper used below on all fsdp modules."""
      frozen_params = []
      for p in fsdp_module.full_params:
        if not p.requires_grad:
          frozen_params.append(p)
        if hasattr(p, "_shard_bwd_hook"):
          assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
          p._shard_bwd_hook[1].remove()
          delattr(p, "_shard_bwd_hook")
      # Free the full params with `requires_grad==False`
      if frozen_params:
        fsdp_module._free_full_params(
            frozen_params,
            apply_opt_barrier=self.optimization_barrier_in_backward)

    # Update root and nested FSDP's hooks and flags.
    for m in self.modules():  # includes self
      if isinstance(m, XlaFullyShardedDataParallel):
        _finalize_parameters(m)
        if not m._pre_backward_hook_has_run:
          m.assert_state(TrainingState.IDLE)
          # The module won't trigger post_backward_hook, so we free the
          # full params here.
          m._free_full_params(
              m.full_params,
              apply_opt_barrier=self.optimization_barrier_in_backward)
        elif any(p.requires_grad for p in m.parameters()):
          # Check if the module has params and if any of them has
          # the `requires_grad` field set. If `requires_grad=False` for
          # all the params, the post_backward hook will not fire and the
          # state will remain in `TrainingState.BACKWARD_PRE`.
          if any([p.requires_grad for p in m.full_params]):
            m.assert_state(TrainingState.BACKWARD_POST)
          else:
            m.assert_state(TrainingState.BACKWARD_PRE)
        else:
          # When `m` and its children has no params or has params but
          # none with `requires_grad==True`, there are two cases:
          # 1. output tensors are `requires_grad==True`. In this case,
          # pre-backward hook is still registered, so it is in BACKWARD_PRE state.
          # 2. output tensors are `requires_grad==False`. In this case,
          # pre-backward hook is not registered, so it is in IDLE state.
          m.assert_state([TrainingState.BACKWARD_PRE, TrainingState.IDLE])

        m.training_state = TrainingState.IDLE
        m._pre_backward_hook_has_run = False
        if m._is_root:
          # reset this flag for cases like "one forward pass + multiple backward passes"
          self._post_backward_callback_queued = False
          # clear this list for next iteration
          assert self._output_pre_backward_hook_registered is not None
          self._output_pre_backward_hook_registered.clear()
          if self.optimization_barrier_in_backward:
            # Ensure that backward pass ops of feature gradients, parameter
            # gradient and sharding, and full-param freeing (which are usually
            # performed in previous modules and are registered to
            # self._backward_opt_barrier_tensors in _grad_opt_barrier_hook,
            # _pre_backward_hook, and _post_backward_hook) are finished before
            # accessing the sharded gradients of this FSDP module.
            params_with_grad = [
                p for p in self._all_sharded_params if p.grad is not None
            ]
            grad_data = [p.grad for p in params_with_grad]
            dependency_tensors = params_with_grad + grad_data
            dependency_tensors.extend(self._backward_opt_barrier_tensors)
            self.optimization_barrier_op(dependency_tensors)
          self._clear_backward_opt_barrier_lists()

    if self.mark_step_on_finalization:
      # Forcing an execution at the end of backward pass to avoid any XLA compiler
      # fusion between backward and optimizer (e.g. AdamW and SGD) step.
      # Here `xm.mark_step` is only called once for the entire backward pass and
      # should therefore only moderately increase the execution time.
      # It may help prevent undesired fusion in backward pass and save more memory.
      if self._debug_print:
        xm.master_print(
            f"mark_step called in FSDP _wait_for_post_backward (_debug_msg: {self._debug_msg})",
            flush=True,
        )
      xm.mark_step()

  @torch.no_grad()
  def _rebuild_full_params(self,
                           dependency_tensors: Optional[List[
                               torch.Tensor]] = None,
                           apply_opt_barrier: bool = True) -> None:
    """
    Gather all shards of params. If `dependency_tensors` is provided,
    it ensures that previous ops to compute tensors in `dependency_tensors`
    are finished before rebuilding the full parameters.

    Note, this is idempotent if full params are already gathered. Callers
    assume the idempotency. So please keep it that way.
    """
    if self.has_full_params:
      return
    if dependency_tensors is None:
      dependency_tensors = []

    if apply_opt_barrier:
      self._apply_opt_barrier_to_params_and_tensors(
          [p for p in self.full_params if p._has_full_param],
          self.sharded_params, dependency_tensors)

    if self.coalesce_all_gather_ops:
      p_to_rebuild, shards_to_all_gather = [], []
    for p, p_shard in zip(self.full_params, self.sharded_params):
      if not p._has_full_param:
        p_shard_data = p_shard
        if apply_opt_barrier:
          self.optimization_barrier_op([p_shard_data])
        if p_shard_data.dtype != self.compute_dtype:
          p_shard_data = p_shard_data.to(self.compute_dtype)
        if self._shard_param_on_dim_0 or self._shard_size_multiple == 1:
          if self.coalesce_all_gather_ops:
            p_to_rebuild.append((p, p_shard))
            shards_to_all_gather.append(p_shard_data)
          else:
            p_padded = self.all_gather_op(
                p_shard_data, groups=self.sharding_groups)
        else:
          # gather full parameter from shards
          # reshape sharded parameters to 2d tensors for efficient gathering on
          # TPUs (see https://github.com/pytorch/xla/issues/3510 for details).
          p_shard_2d = p_shard_data.view(-1, self._shard_size_multiple)
          p_padded = self.all_gather_op(
              p_shard_2d, groups=self.sharding_groups).flatten()
        if not self.coalesce_all_gather_ops:
          if apply_opt_barrier:
            self.optimization_barrier_op([p_padded])
          with torch.autograd._unsafe_preserve_version_counter(p):
            if self._shard_param_on_dim_0:
              if XLA_DISABLE_FUNCTIONALIZATION:
                p.data = p_padded[:p_shard._orig_size[
                    0]]  # Old behavior before Functionalization.
              else:
                torch_xla._XLAC._replace_xla_tensor(
                    p, p_padded[:p_shard._orig_size[0]])
            else:
              if XLA_DISABLE_FUNCTIONALIZATION:
                p.data = p_padded[:p_shard._orig_size.numel()].view(
                    p_shard._orig_size
                )  # Old behavior before Functionalization.
              else:
                torch_xla._XLAC._replace_xla_tensor(
                    p, p_padded[:p_shard._orig_size.numel()].view(
                        p_shard._orig_size))
        p._has_full_param = True

    if self.coalesce_all_gather_ops:
      p_padded_list = self.all_gather_op(
          shards_to_all_gather, groups=self.sharding_groups)
      if apply_opt_barrier:
        self.optimization_barrier_op(p_padded_list)
      for (p, p_shard), p_padded in zip(p_to_rebuild, p_padded_list):
        p.data = p_padded[:p_shard._orig_size[0]]

    self.has_full_params = True

  @torch.no_grad()
  def _free_full_params(self,
                        params: Optional[List[Parameter]] = None,
                        dependency_tensors: Optional[List[torch.Tensor]] = None,
                        apply_opt_barrier: bool = True) -> None:
    """
    Free up storage for full parameters. If `dependency_tensors` is provided,
    it ensures that the full parameters are freed before any new ops that
    depend on tensors in `dependency_tensors` can be executed.
    """
    if params is None:
      full_params = self.full_params
      sharded_params = self.sharded_params
    else:
      full_params = params
      sharded_params = [
          p._sharded_param for p in params if hasattr(p, "_sharded_param")
      ]
    if dependency_tensors is None:
      dependency_tensors = []

    self.has_full_params = False
    for p in full_params:
      if p._has_full_param:
        # free the original full parameter
        with torch.autograd._unsafe_preserve_version_counter(p):
          if XLA_DISABLE_FUNCTIONALIZATION:
            p.data = self._dummy_data_placeholder  # Old behavior before Functionalization.
          else:
            torch_xla._XLAC._replace_xla_tensor(p, self._dummy_data_placeholder)
        p._has_full_param = False

    if apply_opt_barrier:
      self._apply_opt_barrier_to_params_and_tensors(
          [p for p in full_params if p._has_full_param], sharded_params,
          dependency_tensors)

  def _apply_opt_barrier_to_params_and_tensors(
      self, p_list: List[torch.Tensor], p_shard_list: List[torch.Tensor],
      dependency_tensors: List[torch.Tensor]):
    """
    Apply XLA compiler optimization barrier to full and shared parameters
    and other dependency tensors. This is to avoid fusion of the full
    parameter rebuilding and freeing with other computation.

    Otherwise, the XLA compiler might fuse `_rebuild_full_params` and
    `_free_full_params` in the forward pass with any of these calls in the
    backward pass through common subexpression elimination (CSE) and keep the
    full parameters (not freeing them and rebuilding them later, essentially
    changing `reshard_after_forward` to `False` and using more memory).

    This method also introduce control dependency on `dependency_tensors`, so
    that all tensors in `dependency_tensors` must be evaluated before any new
    computation on the full or sharded parameters or `dependency_tensors` can
    happen.
    """
    if len(p_list) + len(p_shard_list) + len(dependency_tensors) == 0:
      return
    self.optimization_barrier_op(p_list + p_shard_list + dependency_tensors)

  def assert_state(self, state: Union[TrainingState,
                                      List[TrainingState]]) -> None:
    """Assert we are in the given state."""
    # Since assert can be turned off and this error checking
    # is really important, we use explicit error checking
    # and raise a ValueError if needed.
    if isinstance(state, TrainingState):
      state = [state]
    if self.training_state not in state:
      msg = f"expected to be in states {state} but current state " f"is {self.training_state}"
      # In case we are failing in the context of autograd hook, asserting
      # may not generate useful msg. So, let's print it to be sure.
      if self.rank == 0:
        print(f"Asserting FSDP instance is: {self}")
        print(f"ERROR: {msg}")
        traceback.print_stack()
      raise ValueError(msg)

  def get_original_names_and_sharded_parameters(self):
    """
    Get the sharded parameters along with their original names. Its output is similar to
    ``named_parameters`` but contains sharded (and flattened) parameters.
    """
    orig_named_parameters = []
    for module_name, m in self.named_modules():  # includes self
      if isinstance(m, XlaFullyShardedDataParallel):
        prefix = "" if module_name == "" else module_name + "."
        for p in self.sharded_params:
          n = prefix + p._orig_name
          n = n.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")
          orig_named_parameters.append((n, p))

    return orig_named_parameters

  def get_shard_metadata(self):
    """
    Get the shard metadata to consolidate the sharded model checkpoints.
    The output from this method should be saved in a checkpoint file and
    used in ``consolidate_sharded_model_checkpoints``.
    """
    shard_info = {}
    flatten_info = {}
    buffer_info = {}
    for module_name, m in self.named_modules(
        remove_duplicate=False):  # includes self
      # remove "_fpw_module." from module names since it is also removed in
      # XlaFullyShardedDataParallel's state_dict()
      module_name = module_name.replace("_fpw_module.", "")

      if isinstance(m, XlaFullyShardedDataParallel):
        sharded_param_info = {}
        for p_shard in m.sharded_params:
          sharded_param_info[p_shard._name] = {
              "_orig_size": p_shard._orig_size,
              "_orig_name": p_shard._orig_name,
          }
        shard_info[module_name] = sharded_param_info

      if isinstance(m, XlaFlattenParamsWrapper):
        for i in range(len(m.flat_params)):
          param_name = f"flat_param_{i}"
          if module_name != "":
            param_name = module_name + "." + param_name
          flatten_info[param_name] = m.metadata(i)

    for name, buf in self.named_buffers():
      if buf is not None and hasattr(buf, "_orig_dtype"):
        buffer_info[name] = {"_orig_dtype": buf._orig_dtype}

    metadata = {
        "shard_info": shard_info,
        "flatten_info": flatten_info,
        "buffer_info": buffer_info,
        "world_size": self.world_size,
        "rank": self.rank,
    }
    return metadata

  def _print_r0(self, msg: str, restart: bool = False) -> None:
    """Debugging utility to print memory usage stats nicely on rank 0"""
    if restart:
      self._tstart = time.time()
    if self.rank == 0:
      memory_info = xm.get_memory_info(xm.xla_device())
      gb_free = memory_info["kb_free"] / 1024 / 1024
      gb_total = memory_info["kb_total"] / 1024 / 1024
      logging.info(
          f"{msg} free={gb_free: .4f} GB, total={gb_total: .4f} GB, t={time.time()-self._tstart: .1f}"
      )

  def _flatten_and_pad_to_world_size(self, tensor: torch.Tensor,
                                     world_size: int) -> torch.Tensor:
    """Flatten and pad a tensor to a given world size (for reduce-scatter)."""
    if self._shard_param_on_dim_0:
      # shard only on dim 0 of the parameter, without flattening
      if tensor.size(0) % world_size != 0:
        pad_size = world_size - tensor.size(0) % world_size
        tensor = F.pad(tensor, [0, 0] * (tensor.dim() - 1) + [0, pad_size])
      return tensor

    tensor = tensor.flatten()
    if tensor.numel() % world_size != 0:
      pad_size = world_size - tensor.numel() % world_size
      tensor = F.pad(tensor, [0, pad_size])

    return tensor

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
      if isinstance(module, XlaFullyShardedDataParallel):
        raise ValueError(
            f"Expected {module_name} to NOT be FullyShardedDataParallel "
            "if using an `auto_wrap_policy`")

    recursive_wrap(**auto_wrap_kwargs, **fsdp_kwargs)


def apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]
) -> Union[torch.Tensor, Dict, List, Tuple, Set]:
  """Recursively apply to all tensor in different kinds of container types."""

  def _apply(
      x: Union[torch.Tensor, Dict, List, Tuple, Set]
  ) -> Union[torch.Tensor, Dict, List, Tuple, Set]:
    if torch.is_tensor(x):
      return fn(x)
    elif isinstance(x, OrderedDict):
      od = x.__class__()
      for key, value in x.items():
        od[key] = _apply(value)
      return od
    elif isinstance(x, PackedSequence):
      _apply(x)
      return x
    elif isinstance(x, dict):
      return {key: _apply(value) for key, value in x.items()}
    elif isinstance(x, list):
      return [_apply(x) for x in x]
    elif isinstance(x, tuple):
      return tuple(_apply(x) for x in x)
    elif isinstance(x, set):
      return {_apply(x) for x in x}
    else:
      return x

  return _apply(container)


def collect_tensors(
    container: Union[torch.Tensor, Dict, List, Tuple,
                     Set]) -> List[torch.Tensor]:
  """Recursively collect to all tensor in different kinds of container types."""

  def _collect(x, out, out_ids) -> None:
    if torch.is_tensor(x):
      if id(x) not in out_ids:
        out_ids.add(id(x))
        out.append(x)
    elif isinstance(x, PackedSequence):
      _collect(x, out, out_ids)
    elif isinstance(x, dict) or isinstance(x, OrderedDict):
      for value in x.values():
        _collect(value, out, out_ids)
    elif isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
      for value in x:
        _collect(value, out, out_ids)

  tensors = []
  _collect(container, tensors, set())
  return tensors


def _calc_grad_norm(parameters: List[torch.nn.Parameter],
                    p: float) -> torch.Tensor:
  """
  Calculate gradient norm of an iterable of parameters.

  Returns:
      Total norm of the parameters (viewed as a single vector).
  """
  if len(parameters) == 0:
    return torch.tensor(0.0)

  if p == inf:
    local_norm = max(par.grad.detach().abs().max() for par in parameters)
  else:
    local_norm = torch.norm(
        torch.stack([torch.norm(par.grad.detach(), p) for par in parameters]),
        p)
  return local_norm


def _cast_floats_tensors(dtype: torch.dtype, *args: Any,
                         **kwargs: Any) -> Tuple[Any, Any]:
  """
  Cast floating point Tensors in *args or **kwargs to dtype if they are not.
  """

  def fn(t):
    if t.dtype != dtype and torch.is_floating_point(t):
      t = t.to(dtype)
    return t

  return apply_to_tensors(fn, args), apply_to_tensors(fn, kwargs)
