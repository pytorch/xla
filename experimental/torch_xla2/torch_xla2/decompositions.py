"""This file contains some decompositons that are not available in torch stable.

Most likely from Content of
https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
at main branch HEAD that we find useful here.

Can also contain decompositions of a torch op in terms of other torch ops.
"""

from typing import Any, Callable, List,  Tuple

import torch
from torch import Tensor
import torch._decomp as decomp
from torch._decomp import decompositions_for_rng
from torch._decomp import register_decomposition
import torch._prims_common as utils
from torch._prims_common.wrappers import out_wrapper


DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

# None of these functions are publicly accessible; get at them
# from torch._decomps
__all__: List[str] = []

aten = torch._ops.ops.aten

def _try_register(op, impl):
    try:
        register_decomposition(op)(impl)
    except:
        pass

@out_wrapper()
def _reflection_pad(a: Tensor, padding: Tuple[int, ...]) -> Tensor:
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return middle - 1 - (middle - 1 - dim_idx.abs()).abs()

    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )

_try_register(aten.reflection_pad1d, _reflection_pad)
_try_register(aten.reflection_pad2d, _reflection_pad)
_try_register(aten.reflection_pad3d, _reflection_pad)

@out_wrapper()
def _replication_pad(a: Tensor, padding: Tuple[int, ...]) -> Tensor:
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return torch.clamp(dim_idx, 0, middle - 1)

    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )

decomp.global_decomposition_table['post_autograd'][aten.replication_pad2d.default] = _replication_pad


def _reflection_or_replication_pad(
    a: Tensor,
    padding: Tuple[int, ...],
    idx_fn: Callable[[int, int, int], Tensor],
) -> Tensor:
    dim = len(padding) // 2
    torch._check(
        a.dim() in (dim + 1, dim + 2),
        lambda: f"reflection_pad{dim}d requires {dim + 1}D or {dim + 2}D input",
    )
    inp_shape = a.shape[-dim:]
    nc_dim = a.dim() - dim

    padding_left = [padding[2 * (dim - 1 - i)] for i in range(dim)]
    padding_right = [padding[2 * (dim - 1 - i) + 1] for i in range(dim)]

    result = a
    for i in range(dim):
        idx: List[Any] = [None] * result.dim()
        idx[i + nc_dim] = idx_fn(padding_left[i], inp_shape[i], padding_right[i])
        result = aten._unsafe_index(result, idx)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(result)
    result = result.contiguous(memory_format=memory_format)
    return result

_try_register(aten.replication_pad1d, _replication_pad)
_try_register(aten.replication_pad3d, _replication_pad)

def bernoulli(self, *, generator=None):
    return (torch.rand_like(self, dtype=torch.float32) < self).to(self.dtype)

_try_register(aten.bernoulli.default, bernoulli)


def rand_like(self, **kwargs):
    dtype = kwargs.get('dtype')
    return torch.rand(self.shape, dtype=dtype)

def channel_shuffle(self, groups):
    batchsize, channels, height, width = self.shape
    channels_per_group = channels // groups
    self = self.reshape(batchsize, groups, channels_per_group, height, width)
    self = self.transpose(1, 2)
    self = self.reshape(batchsize, channels, height, width)
    return self

_try_register(aten.channel_shuffle, channel_shuffle)

_try_register(aten.bernoulli, bernoulli)
_try_register(aten.rand_like, rand_like)

def bernoulli_float(self, p=0.5):
    return self.bernoulli_(torch.tensor(p))

_try_register(aten.bernoulli_.float, bernoulli_float)
_try_register(aten.bernoulli_.Tensor, decompositions_for_rng.bernoulli_)

EXTRA_DECOMP = decomp.get_decompositions([
    torch.ops.aten.upsample_nearest2d,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._adaptive_avg_pool3d,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.native_dropout,
    torch.ops.aten.reflection_pad1d,
    torch.ops.aten.reflection_pad2d,
    torch.ops.aten.reflection_pad3d,
    torch.ops.aten.replication_pad1d,
    torch.ops.aten.replication_pad2d,
    torch.ops.aten.replication_pad3d,
    torch.ops.aten.bernoulli,
    torch.ops.aten.rand_like,
    torch.ops.aten._batch_norm_with_update,
    torch.ops.aten.channel_shuffle,
    torch.ops.aten.nll_loss2d_forward,
    torch.ops.aten.nll_loss2d_backward,
    torch.ops.aten.bernoulli_.Tensor,
    torch.ops.aten.bernoulli_.float,
])
