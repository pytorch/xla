import os
import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

parent_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_folder)


class TestUserComputationDebugCache(unittest.TestCase):

  def setUp(self):
    self.assertTrue(
        os.getenv("XLA_IR_DEBUG") == '1' and os.getenv("XLA_HLO_DEBUG") == '1',
        "XLA_IR_DEBUG and XLA_HLO_DEBUG must be set for this test.",
    )

  def test_user_computation_debug_cache(self):
    """
        Test that user computations with the same IR, but different OpMetadata
        are cached correctly. The metadata is generated when the environment
        variables that enable the Python stack trace for the IR nodes, and
        subsequently, the XLA HLO metadata; `XLA_IR_DEBUG` and `XLA_HLO_DEBUG`
        respectively.
        """

    met.clear_all()

    def fn_op(a, b):
      return xb.Op.tuple([xb.Op.max(a, b) - xb.Op.min(a, b)])

    def input_scope_0(tensor):
      return [torch.sin(tensor), torch.cos(tensor)]

    def input_scope_1(tensor):
      return [torch.sin(tensor), torch.cos(tensor)]

    device = xm.xla_device()
    init_tensor = torch.tensor(10).to(device)

    def create_user_computation(fn):
      inputs = fn(init_tensor)
      comp = xb.create_computation("computation", fn_op,
                                   [xb.tensor_shape(p) for p in inputs])
      _ = torch_xla._XLAC._xla_user_computation("xla::computation", inputs,
                                                comp)
      torch_xla.sync()

    # Create and launch the graph execution with the same IR graph, but with
    # different input tensor scope. When 'XLA_HLO_DEBUG' and 'XLA_IR_DEBUG' are
    # enabled, this will generate different OpMetadata for different input
    # scopes `input_scope_0` and `input_scope_1`, namely `source_line`.
    create_user_computation(input_scope_0)
    create_user_computation(input_scope_1)

    # Ensure that we only compile once, and hit the cache the next time. This
    # is expected as the OpMetadata will not impact the hash of the user
    # computation, as the compiled executable is semantically the same.
    self.assertEqual(met.counter_value("UncachedCompile"), 1)
    self.assertEqual(met.counter_value("CachedCompile"), 1)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
