import sys
import unittest
import torch_xla
import torch
from torch_xla.experimental.scan import scan
from torch.utils._pytree import tree_map, tree_flatten, tree_iter

from test_utils import XlaTestCase


def _loopy_scan(fn, init, xs):
  """A simple scan implemented with for loops serving as reference
  implementation."""
  carry = init
  ys = []
  xs_len = len(next(iter(tree_iter(xs))))
  for i in range(xs_len):
    carry, y = fn(carry, tree_map(lambda x: x[i], xs))
    ys.append(y)
  ys = tree_map(lambda *x: torch.stack(x), *ys)
  return carry, ys


class ScanTest(XlaTestCase):

  def __init__(self, methodName: str = "runTest") -> None:
    super().__init__(methodName)
    self.device = torch_xla.device()

  def compare_pytree(self, expected_pytree, actual_pytree):
    flat_expected_pytree, expected_spec = tree_flatten(expected_pytree)
    flat_actual_pytree, actual_spec = tree_flatten(actual_pytree)
    assert expected_spec == actual_spec
    super().compareResults(flat_expected_pytree, flat_actual_pytree)

  def run_test(self, step_fn, init, xs):
    # Actual output
    final_carry, ys = scan(step_fn, init, xs)
    torch_xla.sync()

    # Expected output
    expected_final_carry, expected_ys = _loopy_scan(step_fn, init, xs)
    torch_xla.sync()

    # Compare
    self.compare_pytree(expected_final_carry, final_carry)
    self.compare_pytree(expected_ys, ys)

    return final_carry, ys

  def test_scan_forward_simple(self):
    """This test uses `scan` to implement `torch.cumsum`."""

    def step_fn(carry, x):
      new_carry = carry + x
      y = new_carry
      return new_carry, y

    init = torch.tensor([0.0, 0.0], device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device)
    final_carry, ys = self.run_test(step_fn, init, xs)

    # Also ensure that our loop-based scan is correct, with manual checks
    # that replicate the step_fn.
    expected_final_carry = torch.sum(xs, dim=0) + init
    expected_ys = torch.cumsum(xs, dim=0)
    self.compare_pytree(expected_final_carry, final_carry)
    self.compare_pytree(expected_ys, ys)

  def test_scan_fn_not_callable(self):
    init = torch.tensor([1.0, 1.0], device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device)
    with self.assertRaises(ValueError):
      scan(1000, init, xs)  # type: ignore

  def test_scan_incompatible_length(self):
    init = torch.tensor([1.0, 1.0], device=self.device)
    xs_1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        device=self.device)
    xs_2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
    with self.assertRaises(ValueError):
      scan(lambda a, b: (a, b), init, (xs_1, xs_2))

  def test_scan_forward_tuples(self):
    """Test scanning over the leading axis of a tuple of tensors simultaneously,
    which is a simple PyTree."""

    def step_fn(carry, x):
      carry1, carry2 = carry
      x1, x2 = x
      new_carry1 = carry1 + x1.sum()
      new_carry2 = carry2 + x2.sum()
      y1 = x1 * 2
      y2 = x2 * 2
      return (new_carry1, new_carry2), (y1, y2)

    init = (torch.tensor([0.0], device=self.device),
            torch.tensor([1.0, 2.0], device=self.device))

    xs = (torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device),
          torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], device=self.device))

    self.run_test(step_fn, init, xs)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
