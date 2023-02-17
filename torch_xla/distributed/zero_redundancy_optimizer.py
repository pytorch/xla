from copy import deepcopy
from typing import (
    Any,
    Iterator,
    Optional,
    Type,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

import torch_xla
import torch_xla.core.xla_model as xm


class ZeroRedundancyOptimizer(Optimizer):
    r"""
    ZeRO-1 wrapper. This class can wraps an arbitrary :class:`optim.Optimizer
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
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        optimizer_class: Type[Optimizer],
        optimizer_dtype: Optional[Any] = None,
        grad_clipping: bool = True,
        max_norm: Optional[float] = None,
        pin_layout: bool = True,
        **defaults: Any,
    ):
        self.params = list(params)
        super().__init__(self.params, defaults)

        self.device = self.params[0].device

        self.rank = xm.get_ordinal()
        self.world_size = xm.xrt_world_size()
        self.cc_op_groups = [list(range(self.world_size))]

        self.optimizer_dtype = optimizer_dtype if optimizer_dtype is not None else torch.float32
        self.grad_clipping = grad_clipping
        self.max_norm = max_norm if max_norm is not None else 1.0
        self.pin_layout = pin_layout

        # Shard parameters for use in optimizer
        self.sharded_params = []
        self._shard_parameters()
        # Optimizer initialization
        self.base_optimizer = optimizer_class(iter(self.sharded_params), **defaults)

    def _shard_tensor(self, tensor: torch.Tensor):
        """
        Get the shard of the input tensor.
        """
        assert tensor.shape[0] % self.world_size == 0, "Not support padding now."
        tensor = tensor.chunk(self.world_size)[self.rank]
        return tensor

    def _shard_parameters(self):
        """
        Shard all parameters.
        """
        xm.unlazy(self.params)
        for param in self.params:
            shard_data = param.data.to(device="cpu")  # move to cpu
            shard_data = self._shard_tensor(shard_data)  # slice it
            if shard_data.dtype != self.optimizer_dtype:
                shard_data = shard_data.to(dtype=self.optimizer_dtype)
            shard_data = shard_data.to(device=self.device)  # move to xla device
            shard = nn.Parameter(shard_data, requires_grad=param.requires_grad)
            self.sharded_params.append(shard)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        Performs a single optimizer step and syncs parameters across all ranks.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reduce full gradients across ranks
        # Assign gradient shards to the respective parameter shards
        for param, shard in zip(self.params, self.sharded_params):
            if param.grad is not None:
                grad_shard = xm.reduce_scatter(
                    xm.REDUCE_SUM,
                    param.grad,
                    scale=1.0 / self.world_size,
                    scatter_dim=0,
                    shard_count=self.world_size,
                    pin_layout=self.pin_layout,
                    groups=self.cc_op_groups,
                )

                if grad_shard.dtype != self.optimizer_dtype:
                    grad_shard = grad_shard.to(dtype=self.optimizer_dtype)
                shard.grad = grad_shard

        if self.grad_clipping:
            # Update unscale/clip with sub partitions
            torch.nn.utils.clip_grad_norm_(self.sharded_params, max_norm=self.max_norm)

        # Step the wrapped optimizer
        loss = self.base_optimizer.step(closure=closure, **kwargs)
        # Remove shards' grads
        self.base_optimizer.zero_grad(set_to_none=True)

        # All gather the new weights across the ranks and assign them to the full parameters
        for param, shard in zip(self.params, self.sharded_params):
            if param.grad is not None:
                shard_data = shard.data
                if param.dtype != self.optimizer_dtype:
                    shard_data = shard_data.to(dtype=param.dtype)
                xm.all_gather(
                    shard_data,
                    dim=0,
                    output=param.data,
                    pin_layout=self.pin_layout,
                    groups=self.cc_op_groups,
                )

        return loss

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['base'] = self.base_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        base = state_dict.pop('base')
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(base)
