
from collections.abc import Sequence
from jax._src.numpy.util import promote_dtypes_inexact
import numpy as np
import jax
from jax import numpy as jnp
from jax._src.util import canonicalize_axis
from jax._src import core
from jax._src.image.scale import _kernels, ResizeMethod
from jax import lax
from typing import Callable

# TODO: This block of code needs to be revisited based on https://github.com/jax-ml/jax/issues/24106
# START ----------------- JAX code copied for fixing scale_and_translate -----------------------------

# JAX Link: https://github.com/jax-ml/jax/blob/18f48bd52abe907ff9818da52f3d195d32910c1b/jax/_src/image/scale.py#L52

def compute_weight_mat(input_size: core.DimSize,
                       output_size: core.DimSize,
                       scale,
                       translation,
                       kernel: Callable,
                       antialias: bool):
  dtype = jnp.result_type(scale, translation)
  inv_scale = 1. / scale
  # When downsampling the kernel should be scaled since we want to low pass
  # filter and interpolate, but when upsampling it should not be since we only
  # want to interpolate.
  kernel_scale = jnp.maximum(inv_scale, 1.) if antialias else 1.
  sample_f = ((jnp.arange(output_size, dtype=dtype) + 0.5) * inv_scale -
              translation * inv_scale - 0.5)
  x = (
      jnp.abs(sample_f[jnp.newaxis, :] -
              jnp.arange(input_size, dtype=dtype)[:, jnp.newaxis]) /
      kernel_scale)
  weights = kernel(x)

  total_weight_sum = jnp.sum(weights, axis=0, keepdims=True)
  weights = jnp.where(
      jnp.abs(total_weight_sum) > 1000. * float(np.finfo(np.float32).eps),
      jnp.divide(weights, jnp.where(total_weight_sum != 0,  total_weight_sum, 1)),
      0)
  # Zero out weights where the sample location is completely outside the input
  # range.
  # Note sample_f has already had the 0.5 removed, hence the weird range below.

  # (barney-s) -------------- returning weights without zeroing ---------------------
  return weights
  input_size_minus_0_5 = core.dimension_as_value(input_size) - 0.5
  return jnp.where(
      jnp.logical_and(sample_f >= -0.5,
                      sample_f <= input_size_minus_0_5)[jnp.newaxis, :], weights, 0)
  # (barney-s) -------------- END returning weights without zeroing ---------------------

# JAX Link: https://github.com/jax-ml/jax/blob/18f48bd52abe907ff9818da52f3d195d32910c1b/jax/_src/image/scale.py#L86

def _scale_and_translate(x, output_shape: core.Shape,
                         spatial_dims: Sequence[int], scale, translation,
                         kernel, antialias: bool, precision):
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  assert len(spatial_dims) == len(scale)
  assert len(spatial_dims) == len(translation)
  if len(spatial_dims) == 0:
    return x
  contractions = []
  in_indices = list(range(len(output_shape)))
  out_indices = list(range(len(output_shape)))
  for i, d in enumerate(spatial_dims):
    d = canonicalize_axis(d, x.ndim)
    m = input_shape[d]
    n = output_shape[d]
    w = compute_weight_mat(m, n, scale[i], translation[i],
                           kernel, antialias).astype(x.dtype)
    contractions.append(w)
    contractions.append([d, len(output_shape) + i])
    out_indices[d] = len(output_shape) + i
  contractions.append(out_indices)
  return jnp.einsum(x, in_indices, *contractions, precision=precision)


# JAX Link: https://github.com/jax-ml/jax/blob/18f48bd52abe907ff9818da52f3d195d32910c1b/jax/_src/image/scale.py#L172

# scale and translation here are scalar elements of an np.array, what is the
# correct type annotation?
def scale_and_translate(image, shape: core.Shape,
                        spatial_dims: Sequence[int],
                        scale, translation,
                        # (barney-s) use string
                        method: str, #(barney-s) | ResizeMethod,
                        antialias: bool = True,
                        precision=lax.Precision.HIGHEST):
  """Apply a scale and translation to an image.

  Generates a new image of shape 'shape' by resampling from the input image
  using the sampling method corresponding to method. For 2D images, this
  operation transforms a location in the input images, (x, y), to a location
  in the output image according to::

    (x * scale[1] + translation[1], y * scale[0] + translation[0])

  (Note the *inverse* warp is used to generate the sample locations.)
  Assumes half-centered pixels, i.e the pixel at integer location ``row, col``
  has coordinates ``y, x = row + 0.5, col + 0.5``, and similarly for other input
  image dimensions.

  If an output location(pixel) maps to an input sample location that is outside
  the input boundaries then the value for the output location will be set to
  zero.

  The ``method`` argument expects one of the following resize methods:

  ``ResizeMethod.LINEAR``, ``"linear"``, ``"bilinear"``, ``"trilinear"``,
    ``"triangle"`` `Linear interpolation`_. If ``antialias`` is ``True``, uses a
    triangular filter when downsampling.

  ``ResizeMethod.CUBIC``, ``"cubic"``, ``"bicubic"``, ``"tricubic"``
    `Cubic interpolation`_, using the Keys cubic kernel.

  ``ResizeMethod.LANCZOS3``, ``"lanczos3"``
    `Lanczos resampling`_, using a kernel of radius 3.

  ``ResizeMethod.LANCZOS5``, ``"lanczos5"``
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  Args:
    image: a JAX array.
    shape: the output shape, as a sequence of integers with length equal to the
      number of dimensions of `image`.
    spatial_dims: A length K tuple specifying the spatial dimensions that the
      passed scale and translation should be applied to.
    scale: A [K] array with the same number of dimensions as image, containing
      the scale to apply in each dimension.
    translation: A [K] array with the same number of dimensions as image,
      containing the translation to apply in each dimension.
    method: the resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC.
    antialias: Should an antialiasing filter be used when downsampling? Defaults
      to ``True``. Has no effect when upsampling.

  Returns:
    The scale and translated image.
  """
  shape = core.canonicalize_shape(shape)
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  if isinstance(method, str):
    method = ResizeMethod.from_string(method)
  if method == ResizeMethod.NEAREST:
    # Nearest neighbor is currently special-cased for straight resize, so skip
    # for now.
    raise ValueError('Nearest neighbor resampling is not currently supported '
                     'for scale_and_translate.')
  assert isinstance(method, ResizeMethod)

  kernel = _kernels[method]
  image, = promote_dtypes_inexact(image)
  scale, translation = promote_dtypes_inexact(scale, translation)
  return _scale_and_translate(image, shape, spatial_dims, scale, translation,
                              kernel, antialias, precision)

# END ----------------- END JAX code copied for testing -----------------------------

