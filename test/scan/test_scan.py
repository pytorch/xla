import sys
import os
import re
import unittest
from functools import reduce

import torch
from functorch.compile import default_partition, min_cut_rematerialization_partition  # type: ignore
from torch.utils._pytree import tree_map, tree_flatten, tree_iter, tree_leaves, PyTree

import torch_xla
import torch_xla.debug.metrics as met
from torch_xla.experimental.scan import scan, value_and_grad_partitioned, tree_flatten_none

parent_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_folder)
from test_utils import XlaTestCase  # type:ignore


def _loopy_scan(fn, init, xs):
  """A simple scan implemented with for loops serving as reference
  implementation."""
  carry = init
  ys = []
  xs_len = len(next(iter(tree_iter(xs))))
  for i in range(xs_len):
    carry, y = fn(carry, tree_map(lambda x: x[i], xs))
    ys.append(y)

  def none_stack(*ys):
    if len(ys) == 0:
      return None
    if ys[0] is None:
      assert all(y is None for y in ys)
      return None
    return torch.stack(ys)

  ys = tree_map(none_stack, *ys)
  return carry, ys


class TestBase(XlaTestCase):

  def setUp(self):
    super().setUp()
    self.device = torch_xla.device()

  def compare_pytree(self, expected_pytree, actual_pytree):
    flat_expected_pytree, expected_spec = tree_flatten(expected_pytree)
    flat_actual_pytree, actual_spec = tree_flatten(actual_pytree)
    assert expected_spec == actual_spec, f"{expected_spec} != {actual_spec}"
    # If there are `None`, they must happen in the same location.
    for expected, actual in zip(flat_expected_pytree, flat_actual_pytree):
      assert (expected is None) == (actual is None), \
            f"Mismatched None. expected: {expected}, actual: {actual}"
    # Get rid of `None` before passing to compareResults.
    flat_expected_pytree = [x for x in flat_expected_pytree if x is not None]
    flat_actual_pytree = [x for x in flat_actual_pytree if x is not None]
    super().compareResults(flat_expected_pytree, flat_actual_pytree)


class ScanTest(TestBase):

  def run_test(self,
               fn,
               init: PyTree,
               xs: PyTree,
               partition_fn=default_partition):
    """Compares the result of scanning with `fn` with our optimized HLO implementation
    against a for loop implementation. Checks both output values and gradients.
    """
    squish = lambda t: reduce(
        lambda a, b: a + b,
        map(lambda v: v.sum()
            if v is not None else 0, tree_leaves(t)), torch.tensor(0.0))
    dupe = lambda v: v.detach().clone().requires_grad_(v.requires_grad)

    # Actual output
    init_scan = tree_map(dupe, init)
    xs_scan = tree_map(dupe, xs)
    final_carry, ys = scan(fn, init_scan, xs_scan, partition_fn=partition_fn)
    # Add up all leaves and `backward()` once.
    (squish(final_carry) + squish(ys)).backward()
    torch_xla.sync()

    # Expected output
    init_loop = tree_map(dupe, init)
    xs_loop = tree_map(dupe, xs)
    expected_final_carry, expected_ys = _loopy_scan(fn, init_loop, xs_loop)
    # Add up all leaves and `backward()` once.
    (squish(expected_final_carry) + squish(expected_ys)).backward()
    torch_xla.sync()

    # Compare values
    self.compare_pytree(expected_final_carry, final_carry)
    self.compare_pytree(expected_ys, ys)

    # Compare gradients
    self.compare_pytree(
        tree_map(lambda v: v.grad, init_loop),
        tree_map(lambda v: v.grad, init_scan))
    self.compare_pytree(
        tree_map(lambda v: v.grad, xs_loop), tree_map(lambda v: v.grad,
                                                      xs_scan))

    return final_carry, ys

  def test_scan_simple(self):
    """This test uses `scan` to implement `torch.cumsum`."""

    def step_fn(carry, x):
      new_carry = carry + x
      y = new_carry
      return new_carry, y

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
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

  def test_scan_tuples(self):
    """Test scanning over the leading axis of a tuple of tensors simultaneously,
    which is a simple PyTree."""

    def fn(carry, x):
      carry1, carry2 = carry
      x1, x2 = x
      new_carry1 = carry1 + x1.sum()
      new_carry2 = carry2 + x2.sum()
      y1 = x1 * 2 + torch.sum(new_carry1)
      y2 = x2 * 2 + torch.sum(new_carry2)
      return (new_carry1, new_carry2), (y1, y2)

    init = (torch.tensor([0.0], requires_grad=True, device=self.device),
            torch.tensor([1.0, 2.0], requires_grad=True, device=self.device))

    xs = (torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                       requires_grad=True,
                       device=self.device),
          torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
                       requires_grad=True,
                       device=self.device))

    self.run_test(fn, init, xs)

  def test_scan_create_tensors(self):
    """Test scanning over a function that internally creates tensors."""

    def fn(carry, x):
      a = torch.tensor([1.0, 2.0], device=self.device)
      b = torch.tensor([3.0, 4.0], device=self.device)
      return carry + a, x + b

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
    self.run_test(fn, init, xs)

  def test_scan_create_tensors_no_transfers_from_device(self):
    """Test that scanning over a function that internally creates tensors
    will not transfer those tensor to host, which can be potentially expensive.
    """

    def fn(carry, x):
      a = torch.tensor([1.0, 2.0], device=self.device)
      b = torch.tensor([3.0, 4.0], device=self.device)
      return carry + a, x + b

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
    met.clear_all()
    self.assertFalse(met.metric_data("TransferFromDeviceTime"))
    # Use `scan` to lower `fn` into HLO and run it. Doing so should not
    # transfer anything from the device to host. `carry` and `ys` still
    # reference data on the device.
    carry, ys = scan(fn, init, xs)
    torch_xla.sync(wait=True)
    self.assertFalse(met.metric_data("TransferFromDeviceTime"))

  def test_scan_internal_in_place_mutation(self):
    """
    Test internal in-place mutations inside the `fn` to be scanned over.
    """

    def fn(carry, x):
      carry = carry.clone()
      carry.add_(x)
      y = x.clone()
      y.add_(42)
      return carry, y

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
    self.run_test(fn, init, xs)

  def test_scan_input_output_aliases_carry(self):
    """
    Test scan still works when a fn output aliases its carry input.
    """

    def fn(carry, x):
      return carry, x + 1

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
    self.run_test(fn, init, xs)

  def test_scan_input_output_aliases_x(self):
    """
    Test scan still works when a fn output aliases its x input.
    """

    def fn(carry, x):
      return carry + 1, x

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
    self.run_test(fn, init, xs)

  def test_scan_input_in_place_mutation(self):
    """
    Test that fn cannot mutate its input. The semantics of that in a `scan`
    is unclear and should be disallowed.
    """

    def fn(carry, x):
      carry.add_(x)
      y = x.clone()
      y.add_(42)
      return carry, y

    init = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                      requires_grad=True,
                      device=self.device)
    with self.assertRaisesRegex(RuntimeError, 'in-place operation'):
      self.run_test(fn, init, xs)

  def test_scan_external_in_place_mutation(self):
    """
    Test that external in-place mutations raise an exception instead of silently
    giving wrong results.
    """
    # TODO(yifeit): Modify this test when external in-place mutation is eventually supported.
    weird_global = torch.tensor([0.0, 0.0], device=torch_xla.device())

    def step_fn(carry, x):
      new_carry = carry + x
      weird_global.add_(1.0)
      y = new_carry + weird_global
      return new_carry, y

    init = torch.tensor([0.0, 0.0], device=torch_xla.device())
    xs = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                      device=torch_xla.device())

    with self.assertRaisesRegex(AssertionError, "FakeTensor"):
      scan(step_fn, init, xs)

  def test_scan_gradness(self):
    """
    Test the gradient output of `scan` when various inputs require or doesn't
    require gradients.
    """

    def test_case(init_requires_grad: bool, xs_requires_grad: bool):

      def fn(carry, x):
        new_carry = carry * x
        y = new_carry + x
        return new_carry, y

      init = torch.tensor([1.0, 1.0],
                          requires_grad=init_requires_grad,
                          device=self.device)
      xs = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                        requires_grad=xs_requires_grad,
                        device=self.device)
      self.run_test(fn, init, xs)

    test_case(True, True)
    test_case(True, False)
    test_case(False, True)

  def test_scan_output_none(self):
    """
    Test scan when `fn` returns `None` as output. This case is exercised by
    `scan_layers`, which only needs the carry.
    """

    def fn(carry, x):
      return torch.cos(carry) + x, None

    init = torch.tensor([1.0, 1.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                      requires_grad=True,
                      device=self.device)
    _final_carry, ys = self.run_test(fn, init, xs)
    self.assertIsNone(ys)

  def test_scan_output_unit(self):
    """
    Test scan when `fn` returns `()` as output.
    """

    def fn(carry, x):
      return torch.cos(carry) + x, ()

    init = torch.tensor([1.0, 1.0], requires_grad=True, device=self.device)
    xs = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                      requires_grad=True,
                      device=self.device)
    _final_carry, ys = self.run_test(fn, init, xs)
    self.assertEqual(ys, ())

  def test_scan_rand_in_fn(self):
    """
    Test that the RNG state in each iteration of `fn` is not the same.
    """

    def step_fn(carry, x):
      new_carry = carry + x
      y = new_carry + torch.rand(2, device=torch_xla.device())
      return new_carry, y

    init = torch.tensor([0.0, 0.0], device=torch_xla.device())
    xs = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                      device=torch_xla.device())
    _, ys = scan(step_fn, init, xs)
    # ys should be a 2D tensor with this shape.
    self.assertEqual(ys.shape, (3, 2))
    # Values across the first dimension should not be the same.
    self.assertNotEqual(ys[0][0], ys[1][0])
    self.assertNotEqual(ys[0][1], ys[1][1])

  def test_scan_with_rematerialization(self):
    """
    Test scanning `fn` but also the backward pass recomputes the forward.
    """

    def fn(carry, x):
      for _ in range(10):
        carry = torch.sin(carry)
      for _ in range(10):
        x = torch.sin(x)
      return carry, x

    carry = torch.randn(4, 4, requires_grad=True, device=self.device)
    xs = torch.randn(20, 4, 4, requires_grad=True, device=self.device)

    # Check the gradients and also cross-check with results from a run
    # where we don't have activation checkpointing.
    final_carry_remat, ys_remat = self.run_test(
        fn, carry, xs, partition_fn=min_cut_rematerialization_partition)
    final_carry, ys = self.run_test(fn, carry, xs)
    super().compareResults(final_carry, final_carry_remat)
    super().compareResults(ys, ys_remat)
    torch_xla.sync()

    SINE_OP = re.compile(r" sine\(f32\b")

    def count_number_of_sines(partition_fn):
      """
      Uses `partition_fn` to partition `fn` into forward and backward passes
      while building the scan operation, then counts the number of `sine` HLO
      operators in the joint graph.

      The intention is that if `partition_fn` recomputes some forward ops
      during the backward, we'll see a larger number of `sine` operations since
      `fn` consists of only `torch.sin` in this test.
      """
      own_carry = carry.clone().detach().requires_grad_()
      own_xs = xs.clone().detach().requires_grad_()
      final_carry, ys = scan(fn, own_carry, own_xs, partition_fn=partition_fn)
      torch_xla.sync()
      (torch.sum(final_carry) + torch.sum(ys)).backward()
      assert own_carry.grad is not None
      assert own_xs.grad is not None
      text: str = torch_xla._XLAC._get_xla_tensors_hlo(
          [own_carry.grad, own_xs.grad])
      return len(SINE_OP.findall(text))

    # Check the HLO to verify that `sine(...)` recomputation happens in the backward
    # in the version using `min_cut_rematerialization_partition`, and never happens
    # in the default partition.
    self.assertGreater(
        count_number_of_sines(min_cut_rematerialization_partition), 10)
    self.assertEqual(count_number_of_sines(default_partition), 0)

  def test_scan_different_dtypes(self):
    """Test that the combine function can output different dtypes."""

    def fn(carry, x):
      bf16_value, f32_value = x
      y = (torch.sin(bf16_value), torch.sin(f32_value))
      return torch.sin(carry), y

    init = torch.tensor([0.0, 0.0],
                        requires_grad=True,
                        device=self.device,
                        dtype=torch.float16)
    bf16_xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                           requires_grad=True,
                           device=self.device,
                           dtype=torch.bfloat16)
    f32_xs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                          requires_grad=True,
                          device=self.device,
                          dtype=torch.float32)
    final_carry, ys = self.run_test(fn, init, (bf16_xs, f32_xs))
    bf16_ys, f32_ys = ys
    self.assertEqual(final_carry.dtype, torch.float16)
    self.assertEqual(bf16_ys.dtype, torch.bfloat16)
    self.assertEqual(f32_ys.dtype, torch.float32)


class PyTreeTest(TestBase):

  def test_tree_flatten_none(self):
    pytree = ((1, 2), (None, 3), None)
    flat, unflatten = tree_flatten_none(pytree)
    assert tuple(flat) == (1, 2, 3)
    assert unflatten(flat) == ((1, 2), (None, 3), None)


class ValueAndGradPartitionedTest(TestBase):

  def test_transform_linear_layer(self):

    def fn(carry, x):
      new_carry = carry @ x
      y = new_carry
      return new_carry, y

    init = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                        requires_grad=True,
                        device=self.device)
    xs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                      requires_grad=True,
                      device=self.device)
    forward, backward = value_and_grad_partitioned(fn, init, xs)

    # Forward should return `(new_carry, (y, (carry, x)))`,
    # because `(carry, x)` are the two intermediate activations (primals),
    # and they will be packed alongside the original output `y`.
    out = forward(init, xs[0])
    torch_xla.sync()
    carry = init
    x = xs[0]
    new_carry = init @ x
    y = new_carry
    self.compare_pytree(out, (new_carry, (y, (carry, x))))

    # Backward should take in `(grad_new_carry, (grad_y, (carry, x)))`, and
    # return `(grad_carry, grad_x)`. `(carry, x)` are the two intermediate
    # activations (primals), and they are packed alongside the gradient with
    # respect to y, `grad_y`.
    grad_new_carry = torch.ones_like(new_carry)
    grad_y = torch.ones_like(y)
    out = backward(grad_new_carry, (grad_y, (carry, x)))
    torch_xla.sync()
    grad_carry = (grad_new_carry + grad_y) @ x.T
    grad_x = carry.T @ (grad_new_carry + grad_y)
    self.compare_pytree(out, (grad_carry, grad_x))

  def test_transform_non_trivial_pytree(self):
    """
    `fn` simulates two linear layers operating on two values a and b.
    Test that we can trace `fn` when it uses non-trivial pytree, and
    compare gradients against those from torch.autograd.
    """

    def fn(carry, x):
      weights = x['weights']
      biases = x['biases']
      carry_a = carry['a']
      carry_b = carry['b']
      new_carry_a = torch.sin((carry_a @ weights) + biases)
      new_carry_b = torch.cos((carry_b @ weights) + biases)
      y = torch.sigmoid(new_carry_a + new_carry_b)
      return {'a': new_carry_a, 'b': new_carry_b}, y

    init = {
        'a': torch.randn(2, 3, requires_grad=True, device=self.device),
        'b': torch.randn(2, 3, requires_grad=True, device=self.device)
    }
    x = {
        'weights': torch.randn(3, 3, requires_grad=True, device=self.device),
        'biases': torch.randn(2, 3, requires_grad=True, device=self.device)
    }

    # Get the forward and backward functions using value_and_grad_partitioned
    forward, backward = value_and_grad_partitioned(
        fn, init, tree_map(lambda v: v.unsqueeze(0), x))

    # Run the forward function
    carry_out, (y_out, activations) = forward(init, x)
    torch_xla.sync()

    # Compute expected outputs and gradients using PyTorch autograd
    def compute_outputs_and_gradients(carry, x):
      # Clone inputs to ensure they're independent
      carry = tree_map(lambda v: v.clone().detach().requires_grad_(True), carry)
      x = tree_map(lambda v: v.clone().detach().requires_grad_(True), x)

      # Forward pass
      new_carry, y = fn(carry, x)

      # Run backward to compute gradients.
      out, _ = tree_flatten((new_carry, y))
      torch.autograd.backward(out, tree_map(lambda v: torch.ones_like(v), out))

      # Collect gradients
      grads = {
          'init': tree_map(lambda v: v.grad, carry),
          'x': tree_map(lambda v: v.grad, x),
      }
      outputs = {'carry': new_carry, 'y': y}
      return outputs, grads

    # Compute expected outputs and gradients
    expected_outputs, expected_grads = compute_outputs_and_gradients(init, x)

    # Compare the outputs from the forward function with the expected outputs
    self.compare_pytree(carry_out, expected_outputs['carry'])
    self.compare_pytree(y_out, expected_outputs['y'])

    # Prepare gradients for the backward function
    grad_carry = tree_map(lambda v: torch.ones_like(v), carry_out)
    grad_y = torch.ones_like(y_out)

    # Run the backward function
    grad_init, grad_x = backward(grad_carry, (grad_y, activations))
    torch_xla.sync()

    # Compare the gradients from the backward function with the expected gradients
    self.compare_pytree(grad_init, expected_grads['init'])
    self.compare_pytree(grad_x, expected_grads['x'])


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
