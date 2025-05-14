"""Tests for get/set_mat_mul_precision from init_python_bindings.cpp"""

import sys
import unittest

import torch
import torch_xla
import torch_xla.backends


class TestMatMulPrecisionGetAndSet(unittest.TestCase):

  def setUp(self):
    self._original = torch_xla.backends.get_mat_mul_precision()
    torch.set_printoptions(precision=20)
    torch_xla.sync()

  def tearDown(self):
    torch_xla.backends.set_mat_mul_precision(self._original)
    torch.set_printoptions(profile="default")
    torch_xla.sync()

  def test_set_mat_mul_precision_error(self):
    # Assert
    with self.assertRaises(ValueError):
      # Act
      torch_xla.backends.set_mat_mul_precision('BAD VALUE')

  def test_get_and_set_mat_mul_precision_default(self):
    # Arrange
    torch_xla.backends.set_mat_mul_precision('default')

    # Act
    status = torch_xla.backends.get_mat_mul_precision()

    # Assert
    self.assertEqual(status, 'default')

  def test_get_and_set_mat_mul_precision_high(self):
    # Arrange
    torch_xla.backends.set_mat_mul_precision('high')

    # Act
    status = torch_xla.backends.get_mat_mul_precision()

    # Assert
    self.assertEqual(status, 'high')

  def test_get_and_set_mat_mul_precision_highest(self):
    # Arrange
    torch_xla.backends.set_mat_mul_precision('highest')

    # Act
    status = torch_xla.backends.get_mat_mul_precision()

    # Assert
    self.assertEqual(status, 'highest')


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
