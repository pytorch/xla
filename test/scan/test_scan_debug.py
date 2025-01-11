import sys
import os
import unittest

import torch
import torch_xla
import torch_xla.debug.metrics as met
from torch_xla.experimental.scan import scan

parent_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_folder)
from test_utils import XlaTestCase  # type:ignore


class ScanDebugTest(XlaTestCase):

  def test_scan_no_recompile_with_debug_annotations(self):
    """
    When someone adds debugging annotations to the HLO via env vars, the
    HLO graph of the combine function captured by scan would have additional metadata
    such as line numbers and scopes. Still, that should not cause the final IR
    graph hash to change. This is subtle because the IR of the `scan` operation will
    reference the HLO computation within.
    """
    assert os.environ["XLA_HLO_DEBUG"] == "1"
    met.clear_all()

    def fn(carry, x):
      carry = carry + x
      y = x + 42
      return carry, y

    # fn2 should trace to the same graph despite having different line numbers
    def fn2(carry, x):
      carry = carry + x
      y = x + 42
      return carry, y

    init = torch.tensor([0.0, 0.0],
                        requires_grad=True,
                        device=torch_xla.device())
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=torch_xla.device())

    # Run some graph involving a scan operation two times.
    for i in range(2):
      init.grad = None
      xs.grad = None
      carry, ys = scan(fn, init, xs)
      (carry.sum() + ys.sum()).backward()
      torch_xla.sync()

    # Use a differently named but semantically the same combine function.
    # This should still trace to identical HLO and hence reuse the cache.
    init.grad = None
    xs.grad = None
    carry, ys = scan(fn2, init, xs)
    (carry.sum() + ys.sum()).backward()
    torch_xla.sync()

    # Should only compile once and cache the last two times.
    self.assertEqual(int(met.counter_value("UncachedCompile")), 1)
    self.assertEqual(int(met.counter_value("CachedCompile")), 2)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
