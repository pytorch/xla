"""Tests for get/set_mat_mul_precision from init_python_bindings.cpp"""

import logging
import sys
import unittest

import torch
import torch_xla
import torch_xla.backends


class TestMatMulPrecisionGetAndSet(unittest.TestCase):

  def setUp(self):
    self.logger_name = torch_xla.backends.logger.name
    self._original_precision = torch_xla.backends.get_mat_mul_precision()
    torch.set_printoptions(precision=20)

  def tearDown(self):
    with self.assertLogs(self.logger_name, level=logging.WARNING):
      torch_xla.backends.set_mat_mul_precision(self._original_precision)
    torch.set_printoptions(profile="default")

  def test_set_mat_mul_precision_warning(self):
    # Arrange
    expected = [(f"WARNING:{self.logger_name}:"
                 f"{torch_xla.backends._WARNING_MESSAGE}")]

    # Act
    with self.assertLogs(self.logger_name, level=logging.WARNING) as cm:
      torch_xla.backends.set_mat_mul_precision('default')

      # Assert
      self.assertEqual(expected, cm.output)

  def test_set_mat_mul_precision_error(self):
    # Assert
    with self.assertRaises(ValueError):
      # Act
      with self.assertLogs(self.logger_name, level=logging.WARNING):
        torch_xla.backends.set_mat_mul_precision('BAD VALUE')

  def test_get_and_set_mat_mul_precision_default(self):
    # Arrange
    with self.assertLogs(self.logger_name, level=logging.WARNING):
      torch_xla.backends.set_mat_mul_precision('default')

    # Act
    status = torch_xla.backends.get_mat_mul_precision()

    # Assert
    self.assertEqual(status, 'default')

  def test_get_and_set_mat_mul_precision_high(self):
    # Arrange
    with self.assertLogs(self.logger_name, level=logging.WARNING):
      torch_xla.backends.set_mat_mul_precision('high')

    # Act
    status = torch_xla.backends.get_mat_mul_precision()

    # Assert
    self.assertEqual(status, 'high')

  def test_get_and_set_mat_mul_precision_highest(self):
    # Arrange
    with self.assertLogs(self.logger_name, level=logging.WARNING):
      torch_xla.backends.set_mat_mul_precision('highest')

    # Act
    status = torch_xla.backends.get_mat_mul_precision()

    # Assert
    self.assertEqual(status, 'highest')


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
