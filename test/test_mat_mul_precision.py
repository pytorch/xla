"""Numeric tests for default precision of mat mul."""

import unittest

import torch
import torch_xla
import torch_xla.backends

import test_utils


class TestMatMulPrecision(unittest.TestCase):

  def _make_input(self):
    eye = torch.eye(1024, device='cpu', dtype=torch.float64)
    rand_ = torch.testing.make_tensor((1024, 1024),
                                      dtype=torch.float64,
                                      device="cpu",
                                      low=0.99,
                                      high=1.01)
    return eye * rand_

  # TODO: Figure out why either PT/XLA or unittest
  # is unable to successfully run this test in a parameterized way.
  # https://github.com/pytorch/xla/issues/9129
  @unittest.skipIf(not test_utils.is_on_tpu(), 'Skipping, not on TPU.')
  @unittest.expectedFailure
  def test_all(self):
    # The number of bit of precise mantissa expected in the result.
    parameters = [
        ('highest', 22),
        ('high', 14),
        ('default', 8),
    ]
    # Although pytest has a slightly more elegant parameterized testing function,
    # all TPU tests user unittest.
    for i, (precision, bits) in enumerate(parameters):
      with self.subTest(precision=precision, bits=bits):
        self._test_parameterized(precision, bits)

  @unittest.skipIf(not test_utils.is_on_tpu(), 'Skipping, not on TPU.')
  def test_highest(self):
    self._test_parameterized('highest', 22)

  @unittest.skipIf(not test_utils.is_on_tpu(), 'Skipping, not on TPU.')
  def test_high(self):
    self._test_parameterized('high', 14)

  @unittest.skipIf(not test_utils.is_on_tpu(), 'Skipping, not on TPU.')
  def test_default(self):
    self._test_parameterized('default', 8)

  # DO NOT add epsilons to this test. These tests must be numerically exact.
  def _test_parameterized(self, precision, bits):
    # Arrange
    torch_xla.backends.set_mat_mul_precision(precision)

    # Diagonal matrices force mat mul through MXU
    # but require only one non-zero accumulation.
    x = self._make_input()
    y = self._make_input()
    reference_float64 = torch.matmul(x, y)

    # TODO: Justify this logic. Why isn't it Why is it not
    # 1 - ((2**8 - 1) / 2**8)**2 (equation stated by per TPU expert)?
    widest_atol = torch.tensor(
        -1 + ((2**(bits) + 1) / 2**bits)**2, dtype=torch.float64)

    narrowest_atol = widest_atol / 4.0

    x = x.to(torch.float32).to('xla')
    y = y.to(torch.float32).to('xla')

    # Act
    actual = torch.matmul(x, y).to('cpu').to(torch.float64)

    # Disable rtol, we know exactly the atol for default, high, and highest.
    torch.testing.assert_close(
        actual,
        reference_float64,
        rtol=0.0,
        atol=widest_atol,
    )

    with self.assertRaises(AssertionError):
      torch.testing.assert_close(
          actual,
          reference_float64,
          rtol=0.0,
          atol=narrowest_atol,
      )

    assert not torch.equal(actual, reference_float64), (
        "Actual product and reference product should not be closer than equal, "
        f"but they are: {torch.diag(actual)} == {torch.diag(reference_float64)}"
    )


# There is no main function. This is designed to be run from
# python -m unittest ...
