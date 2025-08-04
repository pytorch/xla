"""This file contains some decompositons that are not available in torch stable.

Most likely from Content of
https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
at main branch HEAD that we find useful here.

Can also contain decompositions of a torch op in terms of other torch ops.
"""

import functools
from typing import Any, Callable, List, Tuple

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


decomp.global_decomposition_table["post_autograd"][
    aten.replication_pad2d.default] = _replication_pad


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
  dtype = kwargs.get("dtype", self.dtype)
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
  return self.bernoulli_(p)


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
      padding_mode in (0, 1, 2), lambda: f"Invalid padding mode {padding_mode}")

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
  def reflect_coordinates(coords: Tensor, twice_low: int,
                          twice_high: int) -> Tensor:
    if twice_low == twice_high:
      return torch.zeros_like(coords)
    coords_min = twice_low / 2
    coords_span = (twice_high - twice_low) / 2
    coords2 = (coords - coords_min).abs()
    extra = torch.fmod(coords2, coords_span)
    flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
    return torch.where(flips & 1 == 0, extra + coords_min,
                       coords_span + coords_min - extra)

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
  assert three == 3, "Last dim of grid must be 3. got {}".format(three)

  def in_bounds_cond(xs: Tensor, ys: Tensor, zs) -> Tensor:
    xcheck = torch.logical_and(0 <= xs, xs < iW)
    ycheck = torch.logical_and(0 <= ys, ys < iH)
    zcheck = torch.logical_and(0 <= zs, zs < iD)
    return torch.logical_and(xcheck, torch.logical_and(ycheck, zcheck))

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
        torch.where(cond, t, 0).view(N, c, oD, oH, oW) for t in (
            xs.to(dtype=torch.int64),
            ys.to(dtype=torch.int64),
            zs.to(dtype=torch.int64),
            ws,
        ))

  def get_summand(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor,
                  w) -> Tensor:
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
    w_nef = (ix - ix_swb) * (iy_swb - iy) * (id_swb - id_)
    w_swf = (ix_neb - ix) * (iy - iy_neb) * (id_neb - id_)
    w_sef = (ix - ix_nwb) * (iy - iy_nwb) * (id_nwb - id_)
    w_nwb = (ix_sef - ix) * (iy_sef - iy) * (id_ - id_sef)
    w_neb = (ix - ix_swf) * (iy_swf - iy) * (id_ - id_swf)
    w_swb = (ix_nef - ix) * (iy - iy_nef) * (id_ - id_nef)
    w_seb = (ix - ix_nwf) * (iy - iy_nwf) * (id_ - id_nwf)

    return _sum_tensors(
        get_summand(ix, iy, id_, w) for (ix, iy, id_, w) in (
            (ix_nwf, iy_nwf, id_nwf, w_nwf),
            (ix_nef, iy_nef, id_nef, w_nef),
            (ix_swf, iy_swf, id_swf, w_swf),
            (ix_sef, iy_sef, id_sef, w_sef),
            (ix_nwb, iy_nwb, id_nwb, w_nwb),
            (ix_neb, iy_neb, id_neb, w_neb),
            (ix_swb, iy_swb, id_swb, w_swb),
            (ix_seb, iy_seb, id_seb, w_seb),
        ))
  else:  # interpolation_mode == 1:  # Nearest
    ix = compute_source_index(x, iW)
    iy = compute_source_index(y, iH)
    iz = compute_source_index(d, iD)

    ix_nearest = ix.round()
    iy_nearest = iy.round()
    iz_nearest = iz.round()

    return get_summand(ix_nearest, iy_nearest, iz_nearest, 1)


DECOMPOSITIONS = decomp.get_decompositions([
    torch.ops.aten.upsample_bicubic2d,
    torch.ops.aten.upsample_nearest1d,
    torch.ops.aten.upsample_nearest2d,
    torch.ops.aten.upsample_nearest3d,
    torch.ops.aten._upsample_nearest_exact1d,
    torch.ops.aten._upsample_nearest_exact2d,
    torch.ops.aten._upsample_nearest_exact3d,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten._native_batch_norm_legit_functional.default,
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
    torch.ops.aten.addcdiv.default,
    torch.ops.aten.addcdiv.out,
    torch.ops.aten.addcdiv_.default,
    torch.ops.aten.addcmul.default,
    torch.ops.aten.addcmul.out,
    torch.ops.aten.addcmul_.default,
    torch.ops.aten.addr.default,
    torch.ops.aten.addr.out,
    torch.ops.aten.affine_grid_generator.default,
    torch.ops.aten.affine_grid_generator.out,
    torch.ops.aten.alias_copy.default,
    torch.ops.aten.alias_copy.out,
    torch.ops.aten.all.default,
    torch.ops.aten.all.dim,
    torch.ops.aten.all.dims,
    torch.ops.aten.all.out,
    torch.ops.aten.all.dims_out,
    torch.ops.aten.all.all_out,
    torch.ops.aten.all.dimname,
    torch.ops.aten.all.dimname_out,
    torch.ops.aten.aminmax.default,
    torch.ops.aten.aminmax.out,
    torch.ops.aten.arange.default,
    torch.ops.aten.arange.start,
    torch.ops.aten.baddbmm.default,
    torch.ops.aten.baddbmm.out,
    torch.ops.aten.binary_cross_entropy.default,
    torch.ops.aten.binary_cross_entropy.out,
    torch.ops.aten.binary_cross_entropy_backward.default,
    torch.ops.aten.binary_cross_entropy_backward.grad_input,
    torch.ops.aten.binary_cross_entropy_with_logits.default,
    torch.ops.aten.binary_cross_entropy_with_logits.out,
    torch.ops.aten.block_diag.default,
    torch.ops.aten.block_diag.out,
    torch.ops.aten.celu.default,
    torch.ops.aten.celu.out,
    torch.ops.aten.celu_.default,
    torch.ops.aten.channel_shuffle.default,
    torch.ops.aten.channel_shuffle.out,
    torch.ops.aten.clamp_max.default,
    torch.ops.aten.clamp_max.Tensor,
    torch.ops.aten.clamp_max.out,
    torch.ops.aten.clamp_max.Tensor_out,
    torch.ops.aten.clamp_min.default,
    torch.ops.aten.clamp_min.Tensor,
    torch.ops.aten.clamp_min.out,
    torch.ops.aten.clamp_min.Tensor_out,
    torch.ops.aten.col2im.default,
    torch.ops.aten.col2im.out,
    torch.ops.aten.count_nonzero.dim_IntList,
    torch.ops.aten.count_nonzero.dim_IntList_out,
    torch.ops.aten.count_nonzero.default,
    torch.ops.aten.count_nonzero.out,
    torch.ops.aten.linalg_cross.default,
    torch.ops.aten.linalg_cross.out,
    torch.ops.aten.cudnn_batch_norm.default,
    torch.ops.aten.cudnn_batch_norm.out,
    torch.ops.aten.cudnn_batch_norm_backward.default,
    torch.ops.aten.cudnn_batch_norm_backward.out,
    torch.ops.aten.miopen_batch_norm_backward.default,
    torch.ops.aten.miopen_batch_norm_backward.out,
    torch.ops.aten.deg2rad.default,
    torch.ops.aten.deg2rad.out,
    torch.ops.aten.deg2rad_.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.diag_embed.default,
    torch.ops.aten.diag_embed.out,
    torch.ops.aten.diagonal_backward.default,
    torch.ops.aten.diagonal_backward.out,
    torch.ops.aten.dot.default,
    torch.ops.aten.dot.out,
    torch.ops.aten.vdot.default,
    torch.ops.aten.vdot.out,
    torch.ops.aten.elu.default,
    torch.ops.aten.elu.out,
    torch.ops.aten.elu_.default,
    torch.ops.aten.elu_backward.default,
    torch.ops.aten.elu_backward.grad_input,
    torch.ops.aten.embedding_dense_backward.default,
    torch.ops.aten.embedding_dense_backward.out,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.empty_like.out,
    torch.ops.aten._euclidean_dist.default,
    torch.ops.aten.expand_copy.default,
    torch.ops.aten.expand_copy.out,
    torch.ops.aten.eye.default,
    torch.ops.aten.eye.m,
    torch.ops.aten.eye.out,
    torch.ops.aten.eye.m_out,
    torch.ops.aten.fill.Scalar,
    torch.ops.aten.fill.Tensor,
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.fill_.Tensor,
    torch.ops.aten.floor_divide.default,
    torch.ops.aten.floor_divide.Scalar,
    torch.ops.aten.floor_divide.out,
    torch.ops.aten.floor_divide.Scalar_out,
    torch.ops.aten.frac.default,
    torch.ops.aten.frac.out,
    torch.ops.aten.frac_.default,
    torch.ops.aten.gelu_.default,
    torch.ops.aten.gelu_backward.default,
    torch.ops.aten.gelu_backward.grad_input,
    torch.ops.aten.glu.default,
    torch.ops.aten.glu.out,
    torch.ops.aten.glu_backward.default,
    torch.ops.aten.glu_backward.grad_input,
    torch.ops.aten.hardshrink.default,
    torch.ops.aten.hardshrink.out,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid.out,
    torch.ops.aten.hardsigmoid_.default,
    torch.ops.aten.hardsigmoid_backward.default,
    torch.ops.aten.hardsigmoid_backward.grad_input,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.hardswish.out,
    torch.ops.aten.hardswish_.default,
    torch.ops.aten.hardswish_backward.default,
    torch.ops.aten.hardswish_backward.out,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.hardtanh_backward.default,
    torch.ops.aten.hardtanh_backward.grad_input,
    torch.ops.aten.heaviside.default,
    torch.ops.aten.heaviside.out,
    torch.ops.aten.heaviside_.default,
    torch.ops.aten.huber_loss.default,
    torch.ops.aten.huber_loss.out,
    torch.ops.aten.huber_loss_backward.default,
    torch.ops.aten.huber_loss_backward.out,
    torch.ops.aten.im2col.default,
    torch.ops.aten.im2col.out,
    torch.ops.aten.index_add.default,
    torch.ops.aten.index_add.out,
    torch.ops.aten.index_add.dimname,
    torch.ops.aten.index_add_.default,
    torch.ops.aten.index_copy.default,
    torch.ops.aten.index_copy.dimname,
    torch.ops.aten.index_copy.out,
    torch.ops.aten.index_copy_.default,
    torch.ops.aten.index_copy_.dimname,
    torch.ops.aten.index_fill.int_Tensor,
    torch.ops.aten.index_fill.int_Scalar,
    torch.ops.aten.index_fill.Dimname_Scalar,
    torch.ops.aten.index_fill.Dimname_Tensor,
    torch.ops.aten.index_fill.int_Scalar_out,
    torch.ops.aten.index_fill.int_Tensor_out,
    torch.ops.aten.index_fill_.int_Tensor,
    torch.ops.aten.index_fill_.int_Scalar,
    torch.ops.aten.index_fill_.Dimname_Scalar,
    torch.ops.aten.index_fill_.Dimname_Tensor,
    torch.ops.aten.isin.Tensor_Tensor,
    torch.ops.aten.isin.Tensor_Tensor_out,
    torch.ops.aten.isin.Tensor_Scalar,
    torch.ops.aten.isin.Tensor_Scalar_out,
    torch.ops.aten.isin.Scalar_Tensor,
    torch.ops.aten.isin.Scalar_Tensor_out,
    torch.ops.aten.isneginf.default,
    torch.ops.aten.isneginf.out,
    torch.ops.aten.isposinf.default,
    torch.ops.aten.isposinf.out,
    torch.ops.aten.leaky_relu_.default,
    torch.ops.aten.leaky_relu_backward.default,
    torch.ops.aten.leaky_relu_backward.grad_input,
    torch.ops.aten.lerp.Scalar,
    torch.ops.aten.lerp.Tensor,
    torch.ops.aten.lerp.Scalar_out,
    torch.ops.aten.lerp.Tensor_out,
    torch.ops.aten.lerp_.Scalar,
    torch.ops.aten.lerp_.Tensor,
    torch.ops.aten.linspace.Tensor_Tensor,
    torch.ops.aten.linspace.Tensor_Scalar,
    torch.ops.aten.linspace.Scalar_Tensor,
    torch.ops.aten.linspace.default,
    torch.ops.aten.linspace.out,
    torch.ops.aten.linspace.Tensor_Tensor_out,
    torch.ops.aten.linspace.Tensor_Scalar_out,
    torch.ops.aten.linspace.Scalar_Tensor_out,
    torch.ops.aten.logaddexp.default,
    torch.ops.aten.logaddexp.out,
    torch.ops.aten.logaddexp2.default,
    torch.ops.aten.logaddexp2.out,
    torch.ops.aten.logit.default,
    torch.ops.aten.logit.out,
    torch.ops.aten.logit_.default,
    torch.ops.aten.logit_backward.default,
    torch.ops.aten.log_sigmoid_backward.default,
    torch.ops.aten.log_sigmoid_backward.grad_input,
    torch.ops.aten.log_sigmoid_forward.default,
    torch.ops.aten.log_sigmoid_forward.output,
    torch.ops.aten._log_softmax_backward_data.default,
    torch.ops.aten._log_softmax_backward_data.out,
    torch.ops.aten.logspace.Tensor_Tensor,
    torch.ops.aten.logspace.Tensor_Scalar,
    torch.ops.aten.logspace.Scalar_Tensor,
    torch.ops.aten.logspace.default,
    torch.ops.aten.logspace.out,
    torch.ops.aten.logspace.Tensor_Tensor_out,
    torch.ops.aten.logspace.Tensor_Scalar_out,
    torch.ops.aten.logspace.Scalar_Tensor_out,
    torch.ops.aten.logsumexp.default,
    torch.ops.aten.masked_fill.Scalar,
    torch.ops.aten.masked_fill.Tensor,
    torch.ops.aten.masked_fill.Scalar_out,
    torch.ops.aten.masked_fill.Tensor_out,
    torch.ops.aten.masked_fill_.Scalar,
    torch.ops.aten.masked_fill_.Tensor,
    torch.ops.aten.mish.default,
    torch.ops.aten.mish.out,
    torch.ops.aten.mish_.default,
    torch.ops.aten.mse_loss.default,
    torch.ops.aten.mse_loss.out,
    torch.ops.aten.mse_loss_backward.default,
    torch.ops.aten.mse_loss_backward.grad_input,
    torch.ops.aten.multi_margin_loss.default,
    torch.ops.aten.multi_margin_loss.out,
    torch.ops.aten.multilabel_margin_loss_forward.default,
    torch.ops.aten.multilabel_margin_loss_forward.output,
    torch.ops.aten.mv.default,
    torch.ops.aten.mv.out,
    torch.ops.aten.mvlgamma.default,
    torch.ops.aten.mvlgamma.out,
    torch.ops.aten.mvlgamma_.default,
    torch.ops.aten.nansum.default,
    torch.ops.aten.nansum.out,
    torch.ops.aten.nan_to_num.default,
    torch.ops.aten.nan_to_num.out,
    torch.ops.aten.nan_to_num_.default,
    torch.ops.aten.native_batch_norm_backward.default,
    torch.ops.aten.native_batch_norm_backward.out,
    torch.ops.aten.native_dropout_backward.default,
    torch.ops.aten.native_dropout_backward.out,
    torch.ops.aten.native_group_norm_backward.default,
    torch.ops.aten.native_group_norm_backward.out,
    torch.ops.aten.native_layer_norm_backward.default,
    torch.ops.aten.native_layer_norm_backward.out,
    torch.ops.aten.new_empty.default,
    torch.ops.aten.new_empty.out,
    torch.ops.aten.new_full.default,
    torch.ops.aten.new_full.out,
    torch.ops.aten.new_ones.default,
    torch.ops.aten.new_ones.out,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.new_zeros.out,
    torch.ops.aten.nll_loss2d_forward.default,
    torch.ops.aten.nll_loss2d_forward.output,
    torch.ops.aten.nll_loss2d_backward.default,
    torch.ops.aten.nll_loss2d_backward.grad_input,
    torch.ops.aten.nll_loss_backward.default,
    torch.ops.aten.nll_loss_backward.grad_input,
    torch.ops.aten.nll_loss_forward.default,
    torch.ops.aten.nll_loss_forward.output,
    torch.ops.aten.norm.Scalar,
    torch.ops.aten.norm.ScalarOpt_dim,
    torch.ops.aten.norm.names_ScalarOpt_dim,
    torch.ops.aten.norm.ScalarOpt_dim_dtype,
    torch.ops.aten.norm.dtype_out,
    torch.ops.aten.norm.out,
    torch.ops.aten.norm.ScalarOpt_dtype,
    torch.ops.aten.norm.ScalarOpt_dtype_out,
    torch.ops.aten.norm.Scalar_out,
    torch.ops.aten.norm.names_ScalarOpt_dim_dtype,
    torch.ops.aten.norm.names_dtype_out,
    torch.ops.aten.norm.names_out,
    torch.ops.aten.ones.default,
    torch.ops.aten.ones_like.default,
    torch.ops.aten.ones_like.out,
    torch.ops.aten.pixel_shuffle.default,
    torch.ops.aten.pixel_shuffle.out,
    torch.ops.aten.pixel_unshuffle.default,
    torch.ops.aten.pixel_unshuffle.out,
    torch.ops.aten._prelu_kernel.default,
    torch.ops.aten._prelu_kernel_backward.default,
    torch.ops.aten._reshape_alias.default,
    torch.ops.aten.rad2deg.default,
    torch.ops.aten.rad2deg.out,
    torch.ops.aten.rad2deg_.default,
    torch.ops.aten.reflection_pad1d.default,
    torch.ops.aten.reflection_pad1d.out,
    torch.ops.aten.reflection_pad1d_backward.default,
    torch.ops.aten.reflection_pad1d_backward.grad_input,
    torch.ops.aten.reflection_pad2d.default,
    torch.ops.aten.reflection_pad2d.out,
    torch.ops.aten.reflection_pad2d_backward.default,
    torch.ops.aten.reflection_pad2d_backward.grad_input,
    torch.ops.aten.reflection_pad3d.default,
    torch.ops.aten.reflection_pad3d.out,
    torch.ops.aten.reflection_pad3d_backward.default,
    torch.ops.aten.reflection_pad3d_backward.grad_input,
    torch.ops.aten.replication_pad1d.default,
    torch.ops.aten.replication_pad1d.out,
    torch.ops.aten.replication_pad2d.default,
    torch.ops.aten.replication_pad2d.out,
    torch.ops.aten.replication_pad3d.default,
    torch.ops.aten.replication_pad3d.out,
    torch.ops.aten.renorm.default,
    torch.ops.aten.renorm.out,
    torch.ops.aten.renorm_.default,
    torch.ops.aten.resize_as.default,
    torch.ops.aten.resize_as.out,
    torch.ops.aten.roll.default,
    torch.ops.aten.roll.out,
    torch.ops.aten.rot90.default,
    torch.ops.aten.rot90.out,
    torch.ops.aten.rrelu_with_noise.default,
    torch.ops.aten.rrelu_with_noise.out,
    torch.ops.aten.rrelu_with_noise_.default,
    torch.ops.aten.rsub.Tensor,
    torch.ops.aten.rsub.Scalar,
    torch.ops.aten.rsub.Tensor_out,
    torch.ops.aten.rsub.Scalar_out,
    torch.ops.aten._safe_softmax.default,
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    torch.ops.aten.select_backward.default,
    torch.ops.aten.select_backward.out,
    torch.ops.aten.select_scatter.default,
    torch.ops.aten.select_scatter.out,
    torch.ops.aten.sgn.default,
    torch.ops.aten.sgn.out,
    torch.ops.aten.sgn_.default,
    torch.ops.aten.sigmoid_backward.default,
    torch.ops.aten.sigmoid_backward.grad_input,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu.out,
    torch.ops.aten.silu_.default,
    torch.ops.aten.silu_backward.default,
    torch.ops.aten.silu_backward.grad_input,
    torch.ops.aten.sinc.default,
    torch.ops.aten.sinc.out,
    torch.ops.aten.sinc_.default,
    torch.ops.aten.slice_backward.default,
    torch.ops.aten.slice_backward.out,
    torch.ops.aten.smooth_l1_loss.default,
    torch.ops.aten.smooth_l1_loss.out,
    torch.ops.aten.smooth_l1_loss_backward.default,
    torch.ops.aten.smooth_l1_loss_backward.grad_input,
    torch.ops.aten.soft_margin_loss.default,
    torch.ops.aten.soft_margin_loss.out,
    torch.ops.aten.soft_margin_loss_backward.default,
    torch.ops.aten.soft_margin_loss_backward.grad_input,
    torch.ops.aten._softmax_backward_data.default,
    torch.ops.aten._softmax_backward_data.out,
    torch.ops.aten.softplus.default,
    torch.ops.aten.softplus.out,
    torch.ops.aten.softplus_backward.default,
    torch.ops.aten.softplus_backward.grad_input,
    torch.ops.aten.softshrink.default,
    torch.ops.aten.softshrink.out,
    torch.ops.aten.special_entr.default,
    torch.ops.aten.special_entr.out,
    torch.ops.aten.special_log_ndtr.default,
    torch.ops.aten.special_log_ndtr.out,
    torch.ops.aten.special_xlog1py.default,
    torch.ops.aten.special_xlog1py.other_scalar,
    torch.ops.aten.special_xlog1py.self_scalar,
    torch.ops.aten.special_xlog1py.out,
    torch.ops.aten.special_xlog1py.self_scalar_out,
    torch.ops.aten.special_xlog1py.other_scalar_out,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes_copy.default,
    torch.ops.aten.split_with_sizes_copy.out,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.std.default,
    torch.ops.aten.std.dim,
    torch.ops.aten.std.correction,
    torch.ops.aten.std.names_dim,
    torch.ops.aten.std.names_out,
    torch.ops.aten.std.out,
    torch.ops.aten.std.correction_out,
    torch.ops.aten.std.correction_names,
    torch.ops.aten.std.correction_names_out,
    torch.ops.aten.std_mean.default,
    torch.ops.aten.std_mean.dim,
    torch.ops.aten.std_mean.correction,
    torch.ops.aten.std_mean.names_dim,
    torch.ops.aten.std_mean.correction_names,
    torch.ops.aten.std_mean.correction_out,
    torch.ops.aten.stack.default,
    torch.ops.aten.stack.out,
    torch.ops.aten.sum.default,
    torch.ops.aten.sum.out,
    torch.ops.aten.t.default,
    torch.ops.aten.t_copy.out,
    torch.ops.aten.t_copy.default,
    torch.ops.aten.take.default,
    torch.ops.aten.take.out,
    torch.ops.aten.tanh_backward.default,
    torch.ops.aten.tanh_backward.grad_input,
    torch.ops.aten.threshold.default,
    torch.ops.aten.threshold.out,
    torch.ops.aten.threshold_.default,
    torch.ops.aten.threshold_backward.default,
    torch.ops.aten.threshold_backward.grad_input,
    torch.ops.aten.trace.default,
    torch.ops.aten.trace.out,
    torch.ops.aten.transpose.int,
    torch.ops.aten.tril.default,
    torch.ops.aten.tril.out,
    torch.ops.aten.tril_.default,
    torch.ops.aten.triu.default,
    torch.ops.aten.triu.out,
    torch.ops.aten.triu_.default,
    torch.ops.aten.unbind.int,
    torch.ops.aten.unbind.Dimname,
    torch.ops.aten.unfold_backward.default,
    torch.ops.aten.unfold_backward.out,
    torch.ops.aten.unfold_copy.default,
    torch.ops.aten.unfold_copy.out,
    torch.ops.aten._unsafe_index.Tensor,
    torch.ops.aten._unsafe_index_put.default,
    torch.ops.aten._unsafe_masked_index.default,
    torch.ops.aten._unsafe_masked_index_put_accumulate.default,
    torch.ops.aten.unsafe_split.Tensor,
    torch.ops.aten.unsafe_split_with_sizes.default,
    torch.ops.aten.unsqueeze_copy.out,
    torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten._unsafe_view.out,
    torch.ops.aten.upsample_linear1d.default,
    torch.ops.aten.upsample_linear1d.out,
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_bilinear2d.default,
    torch.ops.aten.upsample_bilinear2d.out,
    torch.ops.aten.upsample_trilinear3d.vec,
    torch.ops.aten.upsample_trilinear3d.default,
    torch.ops.aten.upsample_trilinear3d.out,
    torch.ops.aten.xlogy.Tensor,
    torch.ops.aten.xlogy.Scalar_Other,
    torch.ops.aten.xlogy.Scalar_Self,
    torch.ops.aten.xlogy.OutTensor,
    torch.ops.aten.xlogy.OutScalar_Self,
    torch.ops.aten.xlogy.OutScalar_Other,
    torch.ops.aten.xlogy_.Tensor,
    torch.ops.aten.xlogy_.Scalar_Other,
    torch.ops.aten.zero.default,
    torch.ops.aten.zero.out,
    torch.ops.aten.zero_.default,
    torch.ops.aten.zeros.default,
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.zeros_like.out,
    torch.ops.aten._chunk_cat.default,
    torch.ops.aten._chunk_cat.out,
    torch.ops.aten._weight_norm_interface.default,
    torch.ops.aten._weight_norm_interface.out,
    torch.ops.aten.__iand__.Tensor,
    torch.ops.aten.__ixor__.Tensor,
    torch.ops.aten.__ilshift__.Tensor,
    torch.ops.aten.__ilshift__.Scalar,
    torch.ops.aten.__irshift__.Tensor,
    torch.ops.aten.__irshift__.Scalar,
    torch.ops.aten.__ior__.Tensor,
])

MUTABLE_DECOMPOSITION = [
    torch.ops.aten.bernoulli_.Tensor,
    torch.ops.aten.bernoulli_.float,
]
