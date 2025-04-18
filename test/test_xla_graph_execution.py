# Parse local options first, and rewrite the sys.argv[].
# We need to do that before import "common", as otherwise we get an error for
# unrecognized arguments.
import argparse
import os
import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import unittest
import test_utils
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=0)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers
print(FLAGS)

XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


class TestXlaGraphExecutionCheckLevel(test_utils.XlaTestCase):

  def test_graph_execution_check_level_disabled(self):
    # Test disabled checking
    print("Test check level disabled.")
    torch_xla._XLAC._set_torch_xla_graph_execution_check_level(0)
    start_time = time.time()
    x = torch.ones(2, device=xm.xla_device())
    self.assertEqual(x[0], 1.0)  # This should trigger the checking

    torch_xla._XLAC._set_torch_xla_graph_execution_check_level(3)
    start_time = time.time()
    x = torch.ones(2, device=xm.xla_device())
    self.assertEqual(x[0], 1.0)  # This should trigger the checking
    print("--- %s seconds ---" % (time.time() - start_time))
    del x

  def test_graph_execution_check_level_warning(self):
    # Test WARNING level
    print("Test check level as warning.")
    torch_xla._XLAC._set_torch_xla_graph_execution_check_level(1)
    start_time = time.time()
    x = torch.ones(2, device=xm.xla_device())
    self.assertEqual(x[0], 1.0)  # This should trigger the checking
    print("--- %s seconds ---" % (time.time() - start_time))
    del x

  def test_graph_execution_check_level_error(self):
    # Test ERROR level
    print(
        "Test check level as runtime error with warning messages before that.")
    torch_xla._XLAC._set_torch_xla_graph_execution_check_level(2)
    start_time = time.time()
    x = torch.ones(2, device=xm.xla_device())
    with self.assertRaises(RuntimeError) as e:
      self.assertEqual(x[0], 1.0)  # This should trigger the checking
    print("--- %s seconds ---" % (time.time() - start_time))
    del x
    print(
        "--- Timers are added for reference. However, the 1st test runs slower due to memory initialization ---"
    )


if __name__ == '__main__':
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(met.metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
