import jax
import jax.numpy as jnp


def cubic_kernel(x, a=-0.75):
  """Cubic kernel with a = -0.75 (PyTorch-like Keys kernel)"""
  absx = jnp.abs(x)
  x2 = absx * absx
  x3 = x2 * absx
  cond1 = (absx <= 1)
  cond2 = (absx > 1) & (absx < 2)
  f1 = (a + 2) * x3 - (a + 3) * x2 + 1
  f2 = a * x3 - 5 * a * x2 + 8 * a * absx - 4 * a
  return jnp.where(cond1, f1, jnp.where(cond2, f2, 0.0))


def compute_contribs(in_size,
                     out_size,
                     scale,
                     support=2.0,
                     align_corners=False,
                     dtype=None):
  if align_corners:
    if out_size == 1:
      in_coords = jnp.zeros((1,), dtype=dtype)
    else:
      in_coords = jnp.linspace(0, in_size - 1, out_size, dtype=dtype)
  else:
    out_coords = jnp.arange(out_size, dtype=dtype) + 0.5
    in_coords = out_coords / scale - 0.5

  left_idx = jnp.floor(in_coords).astype(jnp.int32) - 1
  idxs = left_idx[:, None] + jnp.arange(4)

  dx = in_coords[:, None] - idxs

  weights = cubic_kernel(dx)

  weights = weights / jnp.sum(weights, axis=1, keepdims=True)
  return idxs, weights


def gather_weights(img, idxs, axis):
  """Safely gather with boundary handling"""
  idxs = jnp.clip(idxs, 0, img.shape[axis] - 1)
  return jnp.take(img, idxs, axis=axis)


def interpolate_along_axis_bchw(img, idxs, weights, axis):
  """
    Interpolate along H (axis=2) or W (axis=3) for tensor (B, C, H, W).
    idxs: (out_size, 4) int32 indices
    weights: (out_size, 4) float32 weights
    """
  assert axis in (2, 3), "Axis must be 2 (H) or 3 (W)"
  out_size = idxs.shape[0]
  k = idxs.shape[1]  # Typically 4 for cubic

  # Clip to input bounds
  idxs = jnp.clip(idxs, 0, img.shape[axis] - 1)  # (out_size, 4)

  def gather_and_weight(i):
    idx = idxs[i]  # (4,)
    w = weights[i]  # (4,)

    def gather_one(offset):
      return jnp.take(img, idx[offset], axis=axis)  # shape (B, C, H, W)

    gathered = jnp.stack([gather_one(o) for o in range(k)],
                         axis=0)  # (4, B, C, H, W)
    weighted = jnp.tensordot(w, gathered, axes=(0, 0))  # (B, C, H, W)
    return weighted

  out = jax.vmap(gather_and_weight)(
      jnp.arange(out_size))  # (out_size, B, C, H, W)

  # Move the interpolated axis back into place
  if axis == 2:  # interpolated over H
    return jnp.moveaxis(out, 0, 2)  # (B, C, out_H, W)
  else:  # axis == 3, interpolated over W
    return jnp.moveaxis(out, 0, 3)  # (B, C, H, out_W)


def interpolate_bicubic_no_aa(img, out_h, out_w, align_corners=False):
  h, w = img.shape[-2:]
  if align_corners and out_h > 1:
    scale_y = (h - 1) / (out_h - 1)
  else:
    scale_y = out_h / h

  if align_corners and out_w > 1:
    scale_x = (w - 1) / (out_w - 1)
  else:
    scale_x = out_w / w

  idxs_y, weights_y = compute_contribs(
      h,
      out_h,
      scale_y,
      align_corners=align_corners,
      dtype=img.dtype,
  )
  tmp = interpolate_along_axis_bchw(img, idxs_y, weights_y, axis=2)

  idxs_x, weights_x = compute_contribs(
      w,
      out_w,
      scale_x,
      align_corners=align_corners,
      dtype=img.dtype,
  )
  out = interpolate_along_axis_bchw(tmp, idxs_x, weights_x, axis=3)
  return out
