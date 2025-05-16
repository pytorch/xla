# Parse local options first, and rewrite the sys.argv[].
# We need to do that before import "common", as otherwise we get an error for
# unrecognized arguments.
import argparse
import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest
import test_utils

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=0)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers
print(FLAGS)


class TestXlaGraphExecution(test_utils.XlaTestCase):

  def test_graph_execution_allowed(self):
    torch_xla._XLAC._set_allow_execution(True)
    x = torch.ones(2, device=xm.xla_device())
    self.assertEqual(x[0], 1.0)  # This should trigger the checking
    del x

  def test_graph_execution_disallowed_with_error(self):
    # Test ERROR when graph execution is not allowed
    # Trigger runtime error for unexpected graph execution
    torch_xla._XLAC._set_allow_execution(
        False)  # this flag disallows graph execution
    x = torch.ones(2, device=xm.xla_device())
    with self.assertRaises(RuntimeError) as e:
      self.assertEqual(x[0], 1.0)  # This should trigger the checking
    self.assertIn(
        "Unexpected execution happens inside the compiled function, exiting",
        str(e.exception))
    del x


if __name__ == '__main__':
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  sys.exit(0 if test.result.wasSuccessful() else 1)
