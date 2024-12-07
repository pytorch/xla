"""This file contains some decompositons that are not available in torch stable.

Most likely from Content of
https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
at main branch HEAD that we find useful here.

Can also contain decompositions of a torch op in terms of other torch ops.
"""

import functools
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
    dtype = kwargs.get('dtype', self.dtype)
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



def _sum_tensors(ts) -> Tensor:
    return functools.reduce(torch.add, ts)


@register_decomposition(aten.grid_sampler_3d)
def _grid_sampler_3d(
    a: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> Tensor:
    """References: https://github.com/pytorch/pytorch/blob/06a7dc21c1005750598c37f3adbc031183c74de6/torch/_decomp/decompositions.py#L4075

    The above implement the 2d case.
    """
    _expand_grid = False
    torch._check(
        interpolation_mode in (0, 1),
        lambda: f"Invalid interpolation mode {interpolation_mode}",
    )
    torch._check(
        padding_mode in (0, 1, 2), lambda: f"Invalid padding mode {padding_mode}"
    )

    # a is 5D: [B, C, D, H, W]

    def unnormalize(coords: Tensor, size: int) -> Tensor:
        # Rescale coordinates from [-1, 1] to:
        #   [0, size - 1] if align_corners is True
        #   [-.5, size -.5] if align_corners is False
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    # Reflects coordinates until they fall between low and high (inclusive).
    # The bounds are passed as twice their value so that half-integer values
    # can be represented as ints.
    def reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int) -> Tensor:
        if twice_low == twice_high:
            return torch.zeros_like(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = torch.fmod(coords2, coords_span)
        flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
        return torch.where(
            flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra
        )

    def compute_coordinates(coords: Tensor, size: int) -> Tensor:
        if padding_mode == 0:  # Zero
            return coords
        elif padding_mode == 1:  # Borders
            return torch.clamp(coords, 0, size - 1)
        else:  # padding_mode == 2, Reflection
            if align_corners:
                coords_reflected = reflect_coordinates(coords, 0, 2 * (size - 1))
            else:
                coords_reflected = reflect_coordinates(coords, -1, 2 * size - 1)
            return torch.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords: Tensor, size: int) -> Tensor:
        coords_un = unnormalize(coords, size)
        return compute_coordinates(coords_un, size)

    N, C, iD, iH, iW = a.shape
    _, oD, oH, oW, three = grid.shape
    assert three == 3, 'Last dim of grid must be 3. got {}'.format(three)


    def in_bounds_cond(xs: Tensor, ys: Tensor, zs) -> Tensor:
        xcheck = torch.logical_and(0 <= xs, xs < iW)
        ycheck = torch.logical_and(0 <= ys, ys < iH)
        zcheck = torch.logical_and(0 <= zs, zs < iD)
        return torch.logical_and(
            xcheck, torch.logical_and(ycheck, zcheck)
        )

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1, 1)

    def clip(xs: torch.Tensor, ys: torch.Tensor, zs, ws: torch.Tensor):
        cond = in_bounds_cond(xs, ys, zs)
        # To clip to inside valid coordinates, we map the coordinates
        # to (x, y) = (0, 0) and also set the weight to 0
        # We also change the shape of the tensor to the appropriate one for
        # broadcasting with N_idx, C_idx for the purposes of advanced indexing
        c = C if _expand_grid else 1
        return tuple(
            torch.where(cond, t, 0).view(N, c, oD, oH, oW)
            for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), zs.to(dtype=torch.int64), ws)
        )

    def get_summand(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, w) -> Tensor:
        # Perform clipping, index into input tensor and multiply by weight
        idx_x, idx_y, idx_z, w_ = clip(ix, iy, iz, w)
        return a[N_idx, C_idx, idx_z, idx_y, idx_x] * w_

    x = grid[..., 0]
    y = grid[..., 1]
    d = grid[..., 2]

    if interpolation_mode == 0:  # Bilinear
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)
        id_ = compute_source_index(d, iD)

        ix_nwf, iy_nwf, id_nwf = ix.floor(), iy.floor(), id_.floor()
        ix_nef, iy_nef, id_nef = ix_nwf + 1, iy_nwf, id_nwf
        ix_swf, iy_swf, id_swf = ix_nwf, iy_nwf + 1, id_nwf
        ix_sef, iy_sef, id_sef = ix_nef, iy_swf, id_nwf
        ix_nwb, iy_nwb, id_nwb = ix_nwf, iy_nwf, id_nwf + 1
        ix_neb, iy_neb, id_neb = ix_nef, iy_nef, id_nwf + 1
        ix_swb, iy_swb, id_swb = ix_swf, iy_swf, id_nwf + 1
        ix_seb, iy_seb, id_seb = ix_sef, iy_sef, id_nwf + 1

        w_nwf = (ix_seb - ix) * (iy_seb - iy) * (id_seb - id_)
        w_nef = (ix - ix_swb) * (iy_swb - iy) * (id_swb- id_)
        w_swf = (ix_neb - ix) * (iy - iy_neb) * (id_neb - id_)
        w_sef = (ix - ix_nwb) * (iy - iy_nwb) * (id_nwb - id_)
        w_nwb = (ix_sef - ix) * (iy_sef - iy) * (id_ - id_sef)
        w_neb = (ix - ix_swf) * (iy_swf - iy) * (id_ - id_swf)
        w_swb = (ix_nef - ix) * (iy - iy_nef) * (id_ - id_nef)
        w_seb = (ix - ix_nwf) * (iy - iy_nwf) * (id_ - id_nwf)

        return _sum_tensors(
            get_summand(ix, iy, id_, w)
            for (ix, iy, id_, w) in (
                (ix_nwf, iy_nwf, id_nwf, w_nwf),
                (ix_nef, iy_nef, id_nef, w_nef),
                (ix_swf, iy_swf, id_swf, w_swf),
                (ix_sef, iy_sef, id_sef, w_sef),
                (ix_nwb, iy_nwb, id_nwb, w_nwb),
                (ix_neb, iy_neb, id_neb, w_neb),
                (ix_swb, iy_swb, id_swb, w_swb),
                (ix_seb, iy_seb, id_seb, w_seb),
            )
        )
    else: #interpolation_mode == 1:  # Nearest
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)
        iz = compute_source_index(d, iD)

        ix_nearest = ix.round()
        iy_nearest = iy.round()
        iz_nearest = iz.round()

        return get_summand(ix_nearest, iy_nearest, iz_nearest, 1)

EXTRA_DECOMP = decomp.get_decompositions([
    torch.ops.aten.upsample_bicubic2d,
    torch.ops.aten.upsample_nearest1d,
    torch.ops.aten.upsample_nearest2d,
    torch.ops.aten.upsample_nearest3d,
    torch.ops.aten._upsample_nearest_exact1d,
    torch.ops.aten._upsample_nearest_exact2d,
    torch.ops.aten._upsample_nearest_exact3d,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._adaptive_avg_pool3d,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.grid_sampler_3d,
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
    torch.ops.aten.log_normal,
])
