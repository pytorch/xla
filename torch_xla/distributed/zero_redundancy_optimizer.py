import copy
from typing import (Any, Iterator, Optional, Type, Union, List, Dict)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

import torch_xla
import torch_xla.core.xla_model as xm


class ZeroRedundancyOptimizer(Optimizer):
  r"""
    ZeRO-1 wrapper. This class can wrap an arbitrary :class:`optim.Optimizer
    <torch.optim.Optimizer>` and shards its states across ranks.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        optimizer_dtype (:class:`torch.dtype`, optional): the desired data type
            of optimizer. Default: ``torch.float32``
        grad_clipping (bool, optional): enable (True) or disable (False) grad
            clipping. Default: True
        max_norm (float, optional): max norm of the gradients, effective only
            when ``grad_clipping`` is True. Default: 1.0
        pin_layout (bool, Optional): if ``True``, then pin the layout in the
            collective ops (all_gather and reduce_scatter). See `xm.all_reduce`
            for details on pinning layout. Default: True
        sharding_groups (list, Optional):
          If specified, ZeRO-1 will use this ``sharding_groups`` for all-gather
          and reduce-scatter ops in full parameter construction and gradient
          sharding. This can be useful for mixing ZeRO-1 with model parallelism
          such as Megatron.
        grad_norm_groups (list, Optional):
          If specified, ZeRO-1 will use this ``grad_norm_groups`` for the
          EXTRA all-reduce op in grad norm calculation. This can be model parallel
          groups when mixing ZeRO-1 with model parallelism such as Megatron.
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.

    .. note:: This runs `step` on sharded parameters. This might lead to
        accuracy disparities compared to using original local optimizer. As
        some optimizers (e.g. LAMB) compute global norm and norm for each
        parameter, using sharded parameter results in different norm values.
    """

  def __init__(
      self,
      params: Iterator[Tensor],
      optimizer_class: Type[Optimizer],
      optimizer_dtype: Optional[Any] = None,
      grad_clipping: bool = True,
      max_norm: Optional[float] = None,
      pin_layout: bool = True,
      sharding_groups: Optional[Any] = None,
      grad_norm_groups: Optional[Any] = None,
      lazy_init: bool = False,
      **defaults: Any,
  ):
    super().__init__(params, defaults)

    self.global_world_size = xm.xrt_world_size()
    self.global_rank = xm.get_ordinal()
    self._sharding_groups = [list(range(self.global_world_size))
                            ] if sharding_groups is None else sharding_groups
    self._grad_norm_groups = grad_norm_groups

    self.optimizer_class = optimizer_class
    self.defaults = defaults
    self.optimizer_dtype = optimizer_dtype if optimizer_dtype is not None else torch.float32
    self.grad_clipping = grad_clipping
    self.max_norm = max_norm if max_norm is not None else 1.0
    self.pin_layout = pin_layout

    self.inited = False
    if not lazy_init:
      self.init_zero()

  def init_zero(self):
    self.local_world_size = len(self.sharding_groups[0])
    self.local_rank = self.global_rank // len(self.sharding_groups)
    # Shard parameters for use in optimizer
    sharded_param_groups = self._shard_parameters()
    # Optimizer initialization
    self.base_optimizer = self.optimizer_class(sharded_param_groups,
                                               **self.defaults)
    self._sync_param_groups(self.param_groups, self.base_optimizer.param_groups)
    self.inited = True

  @property
  def sharding_groups(self):
    return self._sharding_groups

  @sharding_groups.setter
  def sharding_groups(self, new_sharding_groups):
    assert not self.inited, "already inited, cannot change sharding_groups"
    self._sharding_groups = new_sharding_groups

  @staticmethod
  def _sync_param_groups(
      src_param_groups: List[Dict[Any, Any]],
      dst_param_groups: List[Dict[Any, Any]],
  ) -> None:
    r"""
      Syncs the attributes from the source parameter groups to the
      destination parameter groups, except the parameters.

      Example attributes include learning rate or scheduler attributes. The
      two parameter groups should have the same length (i.e. same number of
      parameter groups).

      Arguments:
          src_param_groups (list[dict]): parameter groups giving the
              attribute settings to copy.
          dst_param_groups (list[dict]): parameter groups giving the
              attribute settings to set.
      """
    assert len(src_param_groups) == len(dst_param_groups), \
      "Mismatch between number of source and destination parameter groups"
    for src_param_group, dst_param_group in zip(src_param_groups,
                                                dst_param_groups):
      # Sync all attributes except the parameters
      for attr in filter(lambda x: x != "params", src_param_group.keys()):
        dst_param_group[attr] = src_param_group[attr]

  def _pad_to_world_size(self, tensor: torch.Tensor,
                         world_size: int) -> torch.Tensor:
    """Pad a tensor to a given world size (for reduce-scatter)."""
    if tensor.size(0) % world_size != 0:
      pad_size = world_size - tensor.size(0) % world_size
      tensor = F.pad(tensor, [0, 0] * (tensor.dim() - 1) + [0, pad_size])
    return tensor

  def _shard_tensor(self, tensor: torch.Tensor):
    """
    Get the shard of the input tensor.
    """
    tensor = self._pad_to_world_size(tensor, self.local_world_size)
    tensor = tensor.chunk(self.local_world_size)[self.local_rank]
    return tensor

  def _shard_parameters(self):
    """
    Shard all parameters.
    """
    all_params = []
    for param_group in self.param_groups:
      for param in param_group['params']:
        all_params.append(param)

    self.device = all_params[0].device
    xm.unlazy(all_params)

    sharded_params_groups = []
    for param_group in self.param_groups:
      sharded_params = []
      for param in param_group['params']:
        shard_data = param.data.to(device="cpu")  # move to cpu
        shard_data = self._shard_tensor(shard_data)  # slice it
        if shard_data.dtype != self.optimizer_dtype:
          shard_data = shard_data.to(dtype=self.optimizer_dtype)
        shard_data = shard_data.to(device=self.device)  # move to xla device
        shard = nn.Parameter(shard_data, requires_grad=param.requires_grad)
        sharded_params.append(shard)
      sharded_params_group = copy.copy(param_group)
      sharded_params_group['params'] = sharded_params
      sharded_params_groups.append(sharded_params_group)

    return sharded_params_groups

  @torch.no_grad()
  def _calc_grad_norm(
      self,
      norm_type: Union[float, int] = 2.0,
  ) -> torch.Tensor:
    grads_for_norm = []
    for param_group in self.base_optimizer.param_groups:
      for p in param_group['params']:
        if p.grad is not None:
          grads_for_norm.append(p.grad.detach())
    # Norm parameters.
    if norm_type != 2.0:
      raise RuntimeError(f"only norm type 2 is supported, getting {norm_type}")
    total_norm = torch.zeros([], dtype=self.optimizer_dtype, device=self.device)
    for grad in grads_for_norm:
      grad_norm = (grad * grad).sum()
      total_norm += grad_norm
    # All-reduce across data parallel groups
    xm.all_reduce(
        xm.REDUCE_SUM, [total_norm],
        groups=self._sharding_groups,
        pin_layout=self.pin_layout)
    # All-reduce across other parallel groups, usually model parallel groups
    if self._grad_norm_groups is not None:
      xm.all_reduce(
          xm.REDUCE_SUM, [total_norm],
          groups=self._grad_norm_groups,
          pin_layout=self.pin_layout)
    total_norm = torch.pow(total_norm, 1.0 / norm_type)
    return total_norm

  @torch.no_grad()
  def _clip_grad_norm(
      self,
      max_norm: Union[float, int],
      norm_type: Union[float, int] = 2.0,
  ) -> torch.Tensor:
    """
    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = self._calc_grad_norm(norm_type)

    clip_coeff = torch.tensor(
        max_norm, device=self.device) / (
            total_norm + 1e-6)
    clip_value = torch.where(clip_coeff < 1, clip_coeff,
                             torch.tensor(1., device=self.device))
    for param_group in self.base_optimizer.param_groups:
      for p in param_group['params']:
        if p.grad is not None:
          p.grad.detach().mul_(clip_value)

  @torch.no_grad()
  def step(self, closure=None, **kwargs):
    """
    Performs a single optimizer step and syncs parameters across all ranks.
    """
    assert self.inited, "must call init_zero() first"

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    # sync to base optimizer
    self._sync_param_groups(self.param_groups, self.base_optimizer.param_groups)

    # Reduce full gradients across ranks
    # Assign gradient shards to the respective parameter shards
    for param_group, sharded_param_group in zip(
        self.param_groups, self.base_optimizer.param_groups):
      for param, shard in zip(param_group['params'],
                              sharded_param_group['params']):
        if param.grad is not None:
          padded_grad = self._pad_to_world_size(param.grad,
                                                self.local_world_size)
          grad_shard = xm.reduce_scatter(
              xm.REDUCE_SUM,
              padded_grad,
              scale=1.0 / self.local_world_size,
              scatter_dim=0,
              shard_count=self.local_world_size,
              pin_layout=self.pin_layout,
              groups=self.sharding_groups,
          )

          if grad_shard.dtype != self.optimizer_dtype:
            grad_shard = grad_shard.to(dtype=self.optimizer_dtype)
          shard.grad = grad_shard

    if self.grad_clipping:
      # Update unscale/clip with sub partitions
      self._clip_grad_norm(max_norm=self.max_norm)

    # Step the wrapped optimizer
    # Closure already executed, pass none here
    self.base_optimizer.step(closure=None, **kwargs)
    # Remove shards' grads
    self.base_optimizer.zero_grad(set_to_none=True)

    # All gather the new weights across the ranks and assign them to the full parameters
    for param_group, sharded_param_group in zip(
        self.param_groups, self.base_optimizer.param_groups):
      for param, shard in zip(param_group['params'],
                              sharded_param_group['params']):
        if param.grad is not None:
          shard_data = shard.data
          if param.dtype != self.optimizer_dtype:
            shard_data = shard_data.to(dtype=param.dtype)
          padded_param = xm.all_gather(
              shard_data,
              dim=0,
              pin_layout=self.pin_layout,
              groups=self.sharding_groups,
          )
          param.data.copy_(padded_param.data[:param.size(0)])

    # sync back
    self._sync_param_groups(self.base_optimizer.param_groups, self.param_groups)

    return loss

  def state_dict(self):
    state_dict = super().state_dict()
    base_state = self.base_optimizer.state_dict()['state']
    state_dict['base_state'] = base_state
    return state_dict

  def load_state_dict(self, state_dict):
    state_dict = copy.deepcopy(state_dict)
    base_state = state_dict.pop('base_state')
    super().load_state_dict(state_dict)

    # re-init base optimizer to make sure we have right shards
    self.init_zero()

    tmp = self.base_optimizer.state_dict()
    tmp['state'] = base_state
    self.base_optimizer.load_state_dict(tmp)
