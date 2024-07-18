import os
import logging
import io
import unittest
import importlib

from unittest.mock import patch

from torch_xla.experimental.deprecation import deprecated, mark_deprecated


def old_function():
  return False


def new_function():
  return True


@mark_deprecated(new_function)
def old_funtion_to_wrap():
  return False


class TestDepecation(unittest.TestCase):

  def test_map_to_new_func(self):
    this_module = importlib.import_module(__name__)
    old_function = deprecated(this_module, this_module.new_function,
                              "random_old_name")
    with self.assertLogs(level='WARNING') as log:
      result = old_function()
      self.assertIn("random_old_name", log.output[0])
      assert (result)

  @patch('__main__.new_function')
  def test_decorator(self, mock_new_function):
    mock_new_function.__name__ = "new_name"
    mock_new_function.return_value = True
    with self.assertLogs(level='WARNING') as log:

      @mark_deprecated(new_function)
      def function_to_deprecate():
        return False

      result = function_to_deprecate()
      assert (result)
      self.assertIn("function_to_deprecate", log.output[0])
      self.assertIn(mock_new_function.__name__, log.output[0])
      mock_new_function.assert_called_once()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
