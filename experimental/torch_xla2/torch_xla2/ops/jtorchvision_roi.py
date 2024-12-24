import torch
import functools
import einops
import jax
import jax.numpy as jnp
from . import ops_registry


@functools.partial(jax.vmap, in_axes=(0, None, None))
def _get_grid_per_box(box: jnp.ndarray, size: int,
                      sparse: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Obtain a size x size meshgrid inside the given box.

  Args:
    box: XYXY-format boxes of shape (T, 4).
    size: Resolution of the grid.
    sparse: Whether to return sparse meshgrid.

  Returns:
    Two arrays, each has shape (T, size, size) if sparse=False, or
    (T, size, 1) and (T, 1, size) if sparse=True.
  """
  scale_x = size * 1.0 / (box[2] - box[0])
  scale_y = size * 1.0 / (box[3] - box[1])
  return jnp.meshgrid(  # pytype: disable=bad-return-type  # jnp-type
      (jnp.arange(size, dtype=box.dtype) + 0.5) / scale_y + box[1],
      (jnp.arange(size, dtype=box.dtype) + 0.5) / scale_x + box[0],
      indexing="ij",
      sparse=sparse)

def _roi_align_einsum(feature: jnp.ndarray, boxes: jnp.ndarray,
                      output_size: int, sampling_ratio: int) -> jnp.ndarray:
  """An einsum-based implementation of ROIAlign."""
  height, width = feature.shape[:2]
  grid_y, grid_x = _get_grid_per_box(boxes, output_size * sampling_ratio, True)
  grid_y = jnp.squeeze(grid_y, axis=2)  # (T, output_size * sampling_ratio)
  grid_x = jnp.squeeze(grid_x, axis=1)

  def _get_index_and_weights(grid):
    """Computes the 1d index & their weights to be used in interpolation."""
    grid -= 0.5  # Coordinates -> Index
    x0 = jnp.floor(grid)
    x0x1 = jnp.stack([x0, x0 + 1], axis=-1)
    # No need to handle out-of-bounds indices here, because jax.nn.one_hot
    # ensures that out-of-bounds indices are encoded to all-zero vector.
    # This is equivalent to interpolation with zero padding.

    x1_weights = grid - x0
    x0x1_weights = jnp.stack([1 - x1_weights, x1_weights], axis=-1)
    return x0x1, x0x1_weights

  def _get_einsum_weights(grid: jnp.ndarray, size: int) -> jnp.ndarray:
    """Combines the 1d index & their interpolation weights to do einsum.

    Args:
      grid: (T, output_size * sampling_ratio), 1d grid for each box.
      size: the input size.

    Returns:
      A tensor of shape (T, output_size, size), where result[n, i] is a vector
      that determines how much every input contributes to the i-th output of
      boxes[n].
    """
    # Each is (T, output_size * sampling_ratio, 2)
    x0x1, x0x1_weights = _get_index_and_weights(grid)
    x0x1 = einops.rearrange(
        x0x1, "T (o s) two -> T o (s two)", s=sampling_ratio)
    x0x1_weights = einops.rearrange(
        x0x1_weights, "T (o s) two -> T o (s two) 1", s=sampling_ratio)
    # Multiple samples defined by sampling_ratio should be averaged.
    x0x1_weights = x0x1_weights / sampling_ratio

    # (T, output_size, s*2, size)
    x0x1 = jax.nn.one_hot(x0x1, size, dtype=grid.dtype)
    # In 1d case, every output value is interpolated from sampling_ratio*2
    # input values. So we sum the weights over the s*2 dimension.
    return (x0x1 * x0x1_weights).sum(axis=-2)

  # Bilinear interpolation can be done by two 1d interpolations.
  y_weights = _get_einsum_weights(grid_y, height)
  x_weights = _get_einsum_weights(grid_x, width)
  return jnp.einsum(  # pytype: disable=wrong-arg-types  # jnp-type
      "HWc,ThH,TwW->Thwc", feature, y_weights, x_weights, optimize=True)


def roi_align(feature: jnp.ndarray, boxes: jnp.ndarray, output_size: int,
              sampling_ratio: int) -> jnp.ndarray:
  """ROIAlign operation that crops & resample features within the given boxes.

  Args:
    feature: feature of shape (H, W, C).
    boxes: XYXY boxes of shape (T, 4), boxes to crop from feature.
    output_size: Output resolution.
    sampling_ratio: Over-sampling ratio of each output value.

  Returns:
    Output with shape (T, output_size, output_size, C).
  """
  if len(feature.shape) != 3:
    raise ValueError(f"Expect 3d feature in roi_align! Got {feature.shape}")
  if len(boxes.shape) != 2:
    raise ValueError(f"Expect 2d boxes in roi_align! Got {boxes.shape}")
  return _roi_align_einsum(feature, boxes, output_size, sampling_ratio)

try:
  import torch
  import torchvision
  ops_registry.register_torch_dispatch_op(torch.ops.torchvision.roi_align, roi_align)
except ImportError:
  pass