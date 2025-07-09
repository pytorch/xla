from typing import List, Optional, Tuple
import unittest
from unittest.mock import patch

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from torch_xla.experimental.pallas_kernels.quantized_matmul_kernel import (
    quantized_matmul_int8,
    get_tuned_block_sizes,
    TUNED_BLOCK_SIZES,
)
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


def quantize_array(x, n_bits: int = 8, dim: int = -1):
  max_val = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = max_val / int_max
  x_int = jnp.clip(jnp.round((x / scale)), int_min, int_max).astype(jnp.int8)
  return x_int, scale.astype(x.dtype)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class QuantizedMatmulKernelTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest(
          'This kernel requires a Mosaic feature not available for TPU v4 or earlier.'
      )

  def _test_quantized_matmul(self,
                             dtype,
                             bs,
                             n_input_features,
                             n_output_features,
                             quantize_activation,
                             batch_block_size=None,
                             out_block_size=None,
                             in_block_size=None,
                             atol=1.5):

    prng_key = jax.random.key(1234)
    k0, k1 = jax.random.split(prng_key, 2)
    x = jax.random.normal(k0, (bs, n_input_features), dtype=dtype)
    w = jax.random.normal(
        k1, (n_output_features, n_input_features), dtype=dtype)
    x_copy = x.copy()
    w_copy = w.copy()
    q_w, scalar_w = quantize_array(w)
    scalar_w = jnp.squeeze(scalar_w)
    assert scalar_w.shape == (n_output_features,)

    output = quantized_matmul_int8(
        x,
        q_w,
        scalar_w,
        quantize_activation=quantize_activation,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size).block_until_ready()
    expected = jax.lax.dot_general(
        x_copy, w_copy, dimension_numbers=(((1,), (1,)), ((), ())))

    self.assertEqual(output.dtype, expected.dtype)
    self.assertEqual(output.shape, expected.shape)
    self.assertAllClose(output, expected, atol=atol)

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      bs=[128, 256, 512],
      n_input_features=[128, 256, 512],
      n_output_features=[128, 256, 512],
      quantize_activation=[True],
  )
  def test_quantized_matmul_various_input_shapes(self, dtype, bs,
                                                 n_input_features,
                                                 n_output_features,
                                                 quantize_activation):
    self._test_quantized_matmul(
        dtype,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=quantize_activation,
        batch_block_size=128,
        out_block_size=128,
        in_block_size=128)

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      bs=[64, 192],
      n_input_features=[64, 192],
      n_output_features=[64, 192],
      quantize_activation=[True],
  )
  def test_quantized_matmul_unaligned_input_shapes(self, dtype, bs,
                                                   n_input_features,
                                                   n_output_features,
                                                   quantize_activation):
    self._test_quantized_matmul(
        dtype,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=quantize_activation,
        batch_block_size=128,
        out_block_size=128,
        in_block_size=128)

  @patch(
      'torch_xla.experimental.pallas_kernels.quantized_matmul_kernel.get_tpu_version'
  )
  def test_quantized_matmul_retrieve_block_sizes(self, get_tpu_version):
    tpu_version_to_use = 6
    get_tpu_version.return_value = tpu_version_to_use
    key0 = None
    for key, expected_block_sizes in TUNED_BLOCK_SIZES.items():
      if key[0] == tpu_version_to_use:
        key0 = key
        break
    expected_block_sizes = TUNED_BLOCK_SIZES[key0]
    _, bs, n_output_features, n_input_features, activation_dtype, quantize_activation = key0
    actual_block_sizes = get_tuned_block_sizes(TUNED_BLOCK_SIZES, bs,
                                               n_output_features,
                                               n_input_features,
                                               activation_dtype,
                                               quantize_activation)
    assert actual_block_sizes == expected_block_sizes, f"Expected block sizes {expected_block_sizes}, but got {actual_block_sizes} for key {key0}"

  @parameterized.product(
      dtype=[jnp.bfloat16],
      bs=[16],
      n_input_features=[128, 256],
      n_output_features=[128, 256],
      quantize_activation=[True],
  )
  def test_quantized_matmul_use_tuned_block_sizes(self, dtype, bs,
                                                  n_input_features,
                                                  n_output_features,
                                                  quantize_activation):
    with self.assertRaises(AssertionError):
      self._test_quantized_matmul(
          dtype,
          bs,
          n_input_features,
          n_output_features,
          quantize_activation=quantize_activation,
          batch_block_size=None,
          out_block_size=None,
          in_block_size=None,
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
