import jax
import jax.numpy as jnp
from jax import lax

def bilinear_interpolate(
    input,  # [N, C, H, W]
    roi_batch_ind,  # [K]
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
    ymask,  # [K, IY]
    xmask,  # [K, IX]
):
  """
  Performs bilinear interpolation, respecting the provided masks.

  Args:
    input: Input feature map (N, C, H, W).
    roi_batch_ind: Batch indices for each RoI (K).
    y: Vertical sampling coordinates (K, PH, IY).
    x: Horizontal sampling coordinates (K, PW, IX).
    ymask: Mask for valid y coordinates (K, IY).
    xmask: Mask for valid x coordinates (K, IX).

  Returns:
    Interpolated values (K, C, PH, PW, IY, IX).
  """
  _, channels, height, width = input.shape

  # Clamp coordinates to be within the feature map boundaries
  y = jnp.clip(y, 0)
  x = jnp.clip(x, 0)
  y_low = jnp.floor(y).astype(int)
  x_low = jnp.floor(x).astype(int)
  y_high = jnp.where(y_low >= height - 1, height - 1, y_low + 1)
  y_low = jnp.where(y_low >= height - 1, height - 1, y_low)
  y = jnp.where(y_low >= height - 1, y.astype(input.dtype), y)

  x_high = jnp.where(x_low >= width - 1, width - 1, x_low + 1)
  x_low = jnp.where(x_low >= width - 1, width - 1, x_low)
  x = jnp.where(x_low >= width - 1, x.astype(input.dtype), x)

  ly = y - y_low
  lx = x - x_low
  hy = 1.0 - ly
  hx = 1.0 - lx

  def masked_index(y, x):
    """Indexes the input tensor, respecting the masks."""
    if ymask is not None:
      assert xmask is not None
      y = jnp.where(ymask[:, None, :], y, 0)
      x = jnp.where(xmask[:, None, :], x, 0)
    return input[
        roi_batch_ind[:, None, None, None, None, None],
        jnp.arange(channels)[None, :, None, None, None, None],
        y[:, None, :, None, :, None],  # prev [K, PH, IY]
        x[:, None, None, :, None, :],  # prev [K, PW, IX]
    ]  # [K, C, PH, PW, IY, IX]

  v1 = masked_index(y_low, x_low)
  v2 = masked_index(y_low, x_high)
  v3 = masked_index(y_high, x_low)
  v4 = masked_index(y_high, x_high)

  def outer_prod(y, x):
    return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

  w1 = outer_prod(hy, hx)
  w2 = outer_prod(hy, lx)
  w3 = outer_prod(ly, hx)
  w4 = outer_prod(ly, lx)

  val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
  return val


def roi_align(
    input,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width,
    sampling_ratio=-1,
    aligned=False,
):
  """
  Performs RoI Align operation in JAX.

  Args:
    input: Input feature map (N, C, H, W).
    rois: RoIs (K, 5) with batch index in the first column.
    spatial_scale: Spatial scale to map RoI coordinates to input coordinates.
    pooled_height: Height of the output RoI.
    pooled_width: Width of the output RoI.
    sampling_ratio: Number of sampling points.
    aligned: Whether to align the RoI coordinates.

  Returns:
    Pooled RoIs (K, C, pooled_height, pooled_width).
  """
  orig_dtype = input.dtype

  _, _, height, width = input.shape

  ph = jnp.arange(pooled_height)  # [PH]
  pw = jnp.arange(pooled_width)  # [PW]

  roi_batch_ind = rois[:, 0].astype(int)  # [K]
  offset = 0.5 if aligned else 0.0
  roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
  roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
  roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
  roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

  roi_width = roi_end_w - roi_start_w  # [K]
  roi_height = roi_end_h - roi_start_h  # [K]
  if not aligned:
    roi_width = jnp.clip(roi_width, a_min=1.0)  # [K]
    roi_height = jnp.clip(roi_height, a_min=1.0)  # [K]

  bin_size_h = roi_height / pooled_height  # [K]
  bin_size_w = roi_width / pooled_width  # [K]

  exact_sampling = sampling_ratio > 0

  roi_bin_grid_h = sampling_ratio if exact_sampling else jnp.ceil(
      roi_height / pooled_height)  # scalar or [K]
  roi_bin_grid_w = sampling_ratio if exact_sampling else jnp.ceil(
      roi_width / pooled_width)  # scalar or [K]

  if exact_sampling:
    count = max(roi_bin_grid_h * roi_bin_grid_w, 1)  # scalar
    iy = jnp.arange(roi_bin_grid_h)  # [IY]
    ix = jnp.arange(roi_bin_grid_w)  # [IX]
    ymask = None
    xmask = None
  else:
    count = jnp.clip(roi_bin_grid_h * roi_bin_grid_w, a_min=1)  # [K]
    iy = jnp.arange(height)  # [IY]
    ix = jnp.arange(width)  # [IX]
    ymask = iy[None, :] < roi_bin_grid_h[:, None]  # [K, IY]
    xmask = ix[None, :] < roi_bin_grid_w[:, None]  # [K, IX]

  def from_K(t):
    return t[:, None, None]

  y = (
      from_K(roi_start_h)
      + ph[None, :, None] * from_K(bin_size_h)
      + (iy[None, None, :] + 0.5).astype(input.dtype) * from_K(bin_size_h / roi_bin_grid_h)
  )  # [K, PH, IY]
  x = (
      from_K(roi_start_w)
      + pw[None, :, None] * from_K(bin_size_w)
      + (ix[None, None, :] + 0.5).astype(input.dtype) * from_K(bin_size_w / roi_bin_grid_w)
  )  # [K, PW, IX]
  val = bilinear_interpolate(input, roi_batch_ind, y, x, ymask,
                                 xmask)  # [K, C, PH, PW, IY, IX]

  if not exact_sampling:
    val = jnp.where(ymask[:, None, None, None, :, None], val, 0)
    val = jnp.where(xmask[:, None, None, None, None, :], val, 0)

  output = val.sum((-1, -2))  # remove IY, IX ~> [K, C, PH, PW]
  if isinstance(count, jnp.ndarray):
    output = output / count[:, None, None, None]
  else:
    output = output / count

  output = output.astype(orig_dtype)

  return output