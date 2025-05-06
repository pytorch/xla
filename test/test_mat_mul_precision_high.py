"""Numeric tests for high precision of mat mul.

There are three similar test files, suffixed default, high, and highest.
Unfortunately, the precision cannot reliably
be dynamically changed between tensor operations, so
the tests are split into three files to ensure a fresh
environment for each test.
"""

import os
import unittest

import torch
import torch_xla
import torch_xla.backends
import torch_xla.runtime as xr


def _is_on_tpu():
  return 'XRT_TPU_CONFIG' in os.environ or xr.device_type() == 'TPU'


skipIfNotTpu = unittest.skipIf(not _is_on_tpu(), 'Only supported on TPU')


class TestMatMulPrecisionHigh(unittest.TestCase):

  def _make_input(self):
    eye = torch.eye(1024, device='cpu', dtype=torch.float64)
    rand_ = torch.testing.make_tensor((1024, 1024),
                                      dtype=torch.float64,
                                      device="cpu",
                                      low=0.99,
                                      high=1.01)
    return eye * rand_

  # DO NOT add epsilons to this test. These tests must be numerically exact.
  @skipIfNotTpu
  def test_mat_mul_precision_numerics_high(self):
    # Arrange
    torch_xla.backends.set_mat_mul_precision('high')

    # Diagonal matrices force mat mul through MXU
    # but require only one non-zero accumulation.
    x = self._make_input()
    y = self._make_input()
    reference_float64 = torch.matmul(x, y)

    # 14 bits is an estimate of precision in the
    # three pass technique.
    worst_atol = torch.tensor(
        -1 + ((2**14 + 1) / 2**14)**2, dtype=torch.float64)

    x = x.to(torch.float32).to('xla')
    y = y.to(torch.float32).to('xla')

    # Act
    actual = torch.matmul(x, y).to('cpu').to(torch.float64)

    # Disable rtol, we know exactly the atol for default, high, and highest.
    torch.testing.assert_close(
        actual,
        reference_float64,
        rtol=0.0,
        atol=worst_atol,
    )

    assert not torch.equal(actual, reference_float64), (
        "Actual product and reference product should not be equal, "
        f"but they are: {torch.diag(actual)} == {torch.diag(reference_float64)}"
    )


if __name__ == '__main__':
  unittest.main(verbosity=0)
