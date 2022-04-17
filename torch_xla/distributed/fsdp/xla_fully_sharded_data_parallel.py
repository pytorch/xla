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
import torch_xla.core.xla_model as xm

from .xla_flatten_params_wrapper import XlaFlattenParamsWrapper
from .checkpoint_consolidation import consolidate_sharded_model_checkpoints
from .all_gather_via_all_reduce import all_gather_via_all_reduce


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

      my_module = my_module.to(xm.xla_device())
      sharded_module = XlaFullyShardedDataParallel(my_module)
      optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
      x = sharded_module(x, y=3, z=torch.Tensor([1]))
      loss = x.sum()
      loss.backward()
      optim.step()

  It is also possible to shard individual layers separately and have an outer
  wrapper handle any leftover parameters. This can be helpful to further
  reduce TPU memory usage, reduce system memory usage when initializing large
  models and to improve training speed by overlapping the all-gather step
  across the forward pass.

  .. warning::

      The module should be moved to TPU device *before* wrapping it with
      FSDP. For nested FSDP, the inner FSDP modules also need to be on TPU
      before wrapping.

  .. warning::

      The optimizer must be initialized *after* the module has been wrapped,
      since FSDP will shard parameters in-place and this will break any
      previously initialized optimizers.

  .. warning::

      Please use ``optim.step()`` instead of ``xm.optimizer_step(optim)`` for
      optimizer update. The latter averages the gradients across TPUs, which
      is incorrect for FSDP.

  .. warning::

      When saving checkpoints, the training process on each TPU needs to save
      its own (sharded) model and optimizer state_dict. When resuming, all
      training processes need to load their corresponding (sharded) model and
      optimizer state_dict. Use ``consolidate_sharded_model_checkpoints`` to
      build a full model state_dict for the original unwrapped module from
      the sharded model state_dict.

  Args:
      module (nn.Module):
          module to be wrapped with FSDP.
      reshard_after_forward (bool, Optional):
          if ``True``, reshard parameters after the forward pass. This saves
          memory but slows training. This is only relevant when resharding
          individual layers.
      flatten_parameters (bool, Optional):
          if ``True``, flatten parameters into a single contiguous tensor,
          which improves training speed.
      execute_sharding_on_init (bool, Optional):
          if ``True``, immediately execute the parameter sharding via
          `xm.mark_step` to free up the memory of the full parameters.
      optimization_barrier_on_output (bool, Optional):
          if ``True``, apply `xm.optimization_barrier_` on the FSDP module's
          outputs and their gradients. This avoids XLA fusion with subsequent
          computation after the FSDP module and could save additional memory.
      use_all_gather_via_all_reduce (bool, Optional):
          if ``True``, use PyTorch XLA 1.10's all_gather implementation,
          which performs all_gather via padding and all_reduce and avoids
          memory layout error (see https://github.com/pytorch/xla/issues/3423)
          in previous PyTorch XLA versions (before #3423 is resolved).
      mark_step_on_freeing (bool, Optional):
          if ``True``, call `xm.mark_step` upon freeing full parameters.
          When ``reshard_after_forward`` is ``True``, this option avoid XLA
          compiler fusion by forcing an execution to free memory (see
          https://github.com/pytorch/xla/issues/3455#issuecomment-1085448513
          for details). This option may notably increase the execution time
          and trigger frequent compilation, so it should only be used for
          debugging (e.g. memory profiling) and not in real cases.
  """

  def __init__(
      self,
      module: nn.Module,
      reshard_after_forward: bool = True,
      flatten_parameters: bool = True,
      execute_sharding_on_init: bool = True,
      optimization_barrier_on_output: bool = True,
      use_all_gather_via_all_reduce: bool = False,
      mark_step_on_freeing: bool = False,
      _debug_dummy_forward_pass: bool = False,
      _debug_msg: str = "xla_fsdp",
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
    self.rank = xm.get_ordinal()
    self.world_size = xm.xrt_world_size()
    self.reshard_after_forward = self._orig_reshard_after_forward = reshard_after_forward
    self.flatten_parameters = flatten_parameters
    self.optimization_barrier_on_output = optimization_barrier_on_output
    if use_all_gather_via_all_reduce:
      self.all_gather_op = all_gather_via_all_reduce
    else:
      self.all_gather_op = xm.all_gather
    # TODO (ronghanghu): remove when https://github.com/pytorch/xla/issues/3455 is resolved
    # This is a temporary workaround before after we have a mature solution
    # to avoid undesired fusion with XLA compiler optimization barrier (see
    # https://github.com/pytorch/xla/issues/3455#issuecomment-1085448513
    # for details). This workaround notably increases the execution time and
    # may trigger more compilation, so we need a permanent solution to #3455.
    self.mark_step_on_freeing = mark_step_on_freeing
    self._debug_dummy_forward_pass = _debug_dummy_forward_pass
    self._debug_msg = _debug_msg

    self.gradient_predivide_factor: float = self._get_gradient_predivide_factor(
        self.world_size)
    self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

    self.numel_padded_per_param: List[int] = []
    self._tstart = time.time()

    # Only handle params which are not already sharded. This enables
    # sharding individual layers of a Module, with an outer wrapper to
    # shard any leftover parameters.
    param_names = []
    params = []
    for param_name, param in module.named_parameters():
      if not hasattr(param, "_is_sharded"):
        param_names.append(param_name)
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
    del param_names

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

    # Shard module parameters in place
    self._shard_parameters_(params_to_shard)

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
      xm.wait_device_ops()

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
  ) -> torch.Tensor:
    """
    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.

    Args:
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
            for infinity norm.

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
      total_norm = xm.all_reduce(xm.REDUCE_MAX, local_norm)
    else:
      total_norm = xm.all_reduce(xm.REDUCE_SUM, local_norm**norm_type)
      total_norm = total_norm**(1.0 / norm_type)

    # Now multiply each grad by (max_norm/total_norm), same as torch 1.7 https://tinyurl.com/3wtxhhqq)
    clip_coef = torch.clip(max_norm / (total_norm + 1e-6), 0.0, 1.0)
    for p in params_with_grad:
      p.grad.detach().mul_(clip_coef.to(p.grad.device))

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
      # When freeing the full parameters, we point their `.data` to this placeholder
      # (so that the XLA compiler can reuse the memory storage).
      self._dummy_data_placeholder = torch.zeros(
          1, device=params_to_shard[0].device)

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
        if "xla" not in str(p.device):
          raise ValueError(
              "please moved the module to XLA device before wrapping with FSDP")
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

    # deregister the full parameter tensors from their modules (so that they won't
    # appear in the FSDP model's `parameters()` or `named_parameters()` outputs;
    # only the sharded parameters should appear in the FSDP model's `parameters()`)
    for _, m, n in self.full_param_infos:
      assert n in m._parameters
      p = m._parameters.pop(n)
      object.__setattr__(m, n, p)
    for _, _, m, n, shared_m, shared_n in self.shared_full_param_infos:
      assert n in m._parameters
      p = m._parameters.pop(n)
      object.__setattr__(m, n, p)

    # allocate and register new sharded parameters
    self.numel_padded_per_param = []
    self.sharded_params = []
    for p, (module_name, _, n) in zip(self.full_params, self.full_param_infos):
      assert not hasattr(p, "_is_sharded")

      shard_data, num_padded = self._get_shard(p.data)
      p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
      p_shard._is_sharded = True
      p_shard._orig_size = p.data.size()
      p_shard._orig_name = f"{module_name}.{n}"
      p_shard._name = f"_fsdp_shard.{p_shard._orig_name}".replace(
          ".", "_FSDP_SHARD_SEPARATOR_")
      self.register_parameter(p_shard._name, p_shard)
      self.numel_padded_per_param.append(num_padded)
      self.sharded_params.append(p_shard)
      p._sharded_param = p_shard  # add a handle to the sharded parameter
      # Free the full parameter storage (here we free its `.data`) but keep the tensor itself
      # for auto-grad tracing (like `torch.autograd.Variable` before the tensor-variable merge).
      p.data = self._dummy_data_placeholder
      p._has_full_param = False

    assert len(self.numel_padded_per_param) == len(self.full_params)
    assert len(self.sharded_params) == len(self.full_params)

  def _get_shard(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return the local shard of a full tensor."""
    # Shard using torch.chunk to match all-gather/reduce-scatter.
    chunks = list(torch.flatten(tensor).chunk(self.world_size))
    while len(chunks) < self.world_size:
      chunks.append(chunks[0].new_empty(0))

    # Determine number of padding elements.
    num_to_pad = chunks[0].numel() - chunks[self.rank].numel()
    assert num_to_pad >= 0, num_to_pad

    shard = chunks[self.rank].clone()
    if num_to_pad > 0:
      shard = F.pad(shard, [0, num_to_pad])
    return shard, num_to_pad

  def extra_repr(self) -> str:
    repr = (f"world_size={self.world_size}, "
            f"rank={self.rank}, "
            f"flatten_parameters={self.flatten_parameters}, "
            f"reshard_after_forward={self.reshard_after_forward}")
    return repr

  def __getattr__(self, name: str) -> Any:
    """Forward missing attributes to wrapped module."""
    try:
      return super().__getattr__(name)  # defer to nn.Module's logic
    except AttributeError:
      return getattr(self.module, name)

  def __getitem__(self, key: int) -> Any:
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
        avoiding the extra TPU memory overhead, but with the extra
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
    self._output_pre_backward_hook_registered: Optional[List] = None
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
      self._setup_output_hook_list()

    if self._is_root:
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

  def _setup_output_hook_list(self) -> None:
    """set up a list to avoid registering pre-backward hooks incorrectly."""
    assert self._is_root, "This should only be called on the root"
    self._output_pre_backward_hook_registered = []
    for n, m in self.named_modules():
      if n != "" and isinstance(m, XlaFullyShardedDataParallel):
        m._output_pre_backward_hook_registered = self._output_pre_backward_hook_registered

  def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._lazy_init()

    # Start of a forward pass.
    self.training_state = TrainingState.FORWARD

    # All-gather full parameters.
    self._rebuild_full_params()

    # Register backward hooks to reshard params and reduce-scatter grads.
    # These need to be re-registered every forward pass.
    self._register_post_backward_hooks()

    if not self._debug_dummy_forward_pass:
      outputs = self.module(*args, **kwargs)
    else:
      # Run a dummy forward pass by summing the inputs and full parameter.
      # This can be used to debug FSDP parameter memory consumption.
      outputs = self._dummy_forward(*args, **kwargs)

    if self.reshard_after_forward:
      self._free_full_params()
      # Forcing an execution to free the full parameter memory immediately and avoid any XLA compiler
      # fusion (see https://github.com/pytorch/xla/issues/3455#issuecomment-1085448513 for details).
      # This option may notably increase the execution time and trigger frequent compilation,
      # so it should only be used for debugging (e.g. memory profiling) and not in real cases.
      if self.mark_step_on_freeing:
        xm.mark_step()

    if self.optimization_barrier_on_output:
      # Apply XLA compiler optimization barrier to FSDP outputs and their gradients to avoid
      # fusion across FSDP modules (which sometimes results in higher memory consumption).
      outputs = self._register_optimization_barrier_hooks(outputs)

    # Register pre-backward hooks to all-gather the params for the backward
    # pass (if output's grad was needed). This won't register anything if
    # we are in eval mode.
    # Some model does forward pass multiple times, we need to register the
    # pre-backward hook on every output since the last output's hook has to
    # fire first to setup for backward. However, we use ``self._pre_backward_hook_has_run``
    # to prevent repeated overhead from multiple hook callbacks.
    outputs = self._register_pre_backward_hooks(outputs)

    # Done with a forward pass.
    self.training_state = TrainingState.IDLE

    return outputs

  def _dummy_forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    A dummy forward passs with minimal computation that sums all inputs and
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

  def _register_optimization_barrier_hooks(self, outputs: Any) -> Any:
    """
    Apply `xm.optimization_barrier_` to the outputs and their gradients.
    """

    def _optimization_barrier_grad_hook(t_grad):
      xm.optimization_barrier_([t_grad])
      return t_grad

    def _apply_optimization_barrier(t):
      xm.optimization_barrier_([t])
      t.register_hook(_optimization_barrier_grad_hook)
      return t

    outputs = apply_to_tensors(_apply_optimization_barrier, outputs)
    return outputs

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

    def _pre_backward_hook(*unused: Any) -> None:
      # try to queue final backward callback only once for root, so
      # that final backward callback is attached to the outer most
      # backward graph task and called after all the backward
      # calls are completed.
      if self._is_root:
        self._queue_wait_for_post_backward()

      # All-gather full parameters or switching to the full params.
      # Note, ``self._rebuild_full_params`` is idempotent. So in case it is called
      # unnecessarily, it doesn't incur much overhead.
      if self.reshard_after_forward:
        # Forcing an execution to finish all previous ops (such as freeing earlier params and
        # sharding their gradients) before rebuilding the full parameters and avoid any XLA compiler
        # fusion (see https://github.com/pytorch/xla/issues/3455#issuecomment-1085448513 for details).
        # This option may notably increase the execution time and trigger frequent compilation,
        # so it should only be used for debugging (e.g. memory profiling) and not in real cases.
        if self.mark_step_on_freeing:
          xm.mark_step()

        self._rebuild_full_params()

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

    _registered = 0

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
      # We don't register the pre_backward hook on the same tensor that has been
      # returned from an inner FSDP, unless it is the first one.
      nonlocal _registered
      assert self._output_pre_backward_hook_registered is not None
      if t.requires_grad and (_registered == 0 or id(t)
                              not in self._output_pre_backward_hook_registered):
        t.register_hook(_pre_backward_hook)
        self._output_pre_backward_hook_registered.append(id(t))
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

  @torch.no_grad()
  def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
    """
    At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
    full gradient for the local batch. The reduce-scatter op will replace
    ``param.grad`` with a single shard of the summed gradient across all
    TPUs. This shard will align with the current TPU rank. For example::

        before reduce_scatter:
            param.grad (TPU #0): [1, 2, 3, 4]
            param.grad (TPU #1): [5, 6, 7, 8]

        after reduce_scatter:
            param.grad (TPU #0): [6, 8]    # 1+5, 2+6
            param.grad (TPU #1): [10, 12]  # 3+7, 4+8

    The local TPU's ``optim.step`` is responsible for updating a single
    shard of params, also corresponding to the current TPU's rank. This
    alignment is created by :func:`_shard_parameters_`, which ensures that
    the local optimizer only sees the relevant parameter shard.
    """
    # First hook callback will see PRE state. If we have multiple params,
    # then subsequent hook callbacks will see POST state.
    self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
    self.training_state = TrainingState.BACKWARD_POST
    if param.grad is None:
      return

    assert param.grad is not None, param.shape
    if param.grad.requires_grad:
      raise RuntimeError(
          "FSDP only works with gradients that don't require gradients")

    if self._require_backward_grad_sync or self.reshard_after_forward:
      # Free full params. As a special case, we don't free the full params
      # when in a ``no_sync`` context (as inversely indicated by
      # ``self._require_backward_grad_sync``), since the params will not
      # get updated before the next forward. This saves networking
      # bandwidth but uses more TPU memory.
      self._free_full_params([param])

    if not self._require_backward_grad_sync:
      return

    if self.gradient_predivide_factor > 1:
      # Average grad by world_size for consistency with PyTorch DDP.
      param.grad.data.div_(self.gradient_predivide_factor)

    # Shard the gradients with `reduce_scatter`.
    grad = param.grad.data
    # Clear grad on the tensor, so any repeated gradient computations do not interfere with this reduction.
    param.grad = None
    grad_flat = _flatten_and_pad_to_world_size(grad, self.world_size)
    reduced_grad = xm.reduce_scatter(
        xm.REDUCE_SUM,
        grad_flat,
        scale=1.0,
        scatter_dim=0,
        shard_count=self.world_size)
    if self.gradient_postdivide_factor > 1:
      # Average grad by world_size for consistency with PyTorch DDP.
      reduced_grad.data.div_(self.gradient_postdivide_factor)

    # Accumulate into the gradient shard.
    assert hasattr(param, "_sharded_param")
    p_shard = param._sharded_param
    if p_shard.grad is None:
      p_shard.grad = reduced_grad.data
    else:
      assert p_shard.grad.shape == reduced_grad.shape
      assert p_shard.grad.device == reduced_grad.device
      p_shard.grad.data += reduced_grad.data

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
      for p in fsdp_module.full_params:
        if not p.requires_grad:
          continue
        if hasattr(p, "_shard_bwd_hook"):
          assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
          p._shard_bwd_hook[1].remove()
          delattr(p, "_shard_bwd_hook")

    # Update root and nested FSDP's hooks and flags.
    for m in self.modules():  # includes self
      if isinstance(m, XlaFullyShardedDataParallel):
        _finalize_parameters(m)
        m._pre_backward_hook_has_run = False
        if any(p.requires_grad for p in m.parameters()):
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

        if m._is_root:
          # reset this flag for cases like "one forward pass + multiple backward passes"
          self._post_backward_callback_queued = False
          # clear this list for next iteration
          assert self._output_pre_backward_hook_registered is not None
          self._output_pre_backward_hook_registered.clear()

  @torch.no_grad()
  def _rebuild_full_params(self) -> None:
    """
    Gather all shards of params.

    Note, this is idempotent if full params are already gathered. Callers
    assume the idempotency. So please keep it that way.
    """
    if self.has_full_params:
      return
    p_list, p_shard_list, p_data_list, p_shared_data_list = [], [], [], []
    for p, p_shard in zip(self.full_params, self.sharded_params):
      if not p._has_full_param:
        # gather full parameter from shards
        p_padded = self.all_gather_op(p_shard.data).flatten().detach()
        p_data = p_padded[:p_shard._orig_size.numel()].view(p_shard._orig_size)
        p_list.append(p)
        p_shard_list.append(p_shard)
        p_data_list.append(p_data)
        p_shared_data_list.append(p_shard.data)

    if len(p_data_list) + len(p_shared_data_list) > 0:
      # Apply the XLA compiler optimization barrier to avoid fusion of the
      # full parameter reconstruction with other computation.
      # Otherwise, the XLA compiler might fuse `_rebuild_full_params` in the
      # the forward pass with any `_rebuild_full_params` in the backward pass
      # through common subexpression elimination (CSE) and keep the full
      # parameters (not freeing them and rebuilding them later, essentially
      # changing `reshard_after_forward` to `False`` and using more memory).
      xm.optimization_barrier_(p_data_list + p_shared_data_list)
    for p, p_shard, p_data, p_shard_data in zip(p_list, p_shard_list,
                                                p_data_list,
                                                p_shared_data_list):
      p.data = p_data
      p_shard.data = p_shard_data
      p._has_full_param = True
    self.has_full_params = True

  @torch.no_grad()
  def _free_full_params(self, params: Optional[List[Parameter]] = None) -> None:
    """Free up storage for full parameters."""
    if params is None:
      params = self.full_params
    self.has_full_params = False
    p_list, p_data_list = [], []
    for p in params:
      if p._has_full_param:
        # free the original full parameter
        p_data = self._dummy_data_placeholder
        p_list.append(p)
        p_data_list.append(p_data)

    if len(p_data_list) > 0:
      # Apply the XLA compiler optimization barrier to avoid fusion of the
      # full parameter freeing with other computation.
      # Otherwise, the XLA compiler might fuse `_free_full_params` in the
      # forward pass with any `_free_full_params` in the backward pass
      # through common subexpression elimination (CSE) and keep the full
      # parameters (not freeing them and rebuilding them later, essentially
      # changing `reshard_after_forward` to `False`` and using more memory).
      xm.optimization_barrier_(p_data_list)
    for p, p_data in zip(p_list, p_data_list):
      p.data = p_data
      p._has_full_param = False

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
    for module_name, m in self.named_modules():  # includes self
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

    metadata = {
        "shard_info": shard_info,
        "flatten_info": flatten_info,
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


def _flatten_and_pad_to_world_size(tensor: torch.Tensor,
                                   world_size: int) -> torch.Tensor:
  """Flatten and pad a tensor to a given world size (for reduce-scatter)."""
  tensor = tensor.flatten()
  if tensor.numel() % world_size != 0:
    pad_size = world_size - tensor.numel() % world_size
    tensor = F.pad(tensor, [0, pad_size])

  return tensor


def apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple,
                                   Set]) -> Any:
  """Recursively apply to all tensor in different kinds of container types."""

  def _apply(x: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
    if torch.is_tensor(x):
      return fn(x)
    elif isinstance(x, OrderedDict):
      od = x.__class__()
      for key, value in x.items():
        od[key] = _apply(value)
      return od
    elif isinstance(x, PackedSequence):
      _apply(x.data)
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
