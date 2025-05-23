import re
import textwrap
import torch
import torch_xla
import test_utils
import unittest

# Processes a string, so that it can be used as the expected error regex.
# Specifically, it does 3 things:
#
#   1. s[1:]: assumes the first character of the string is a new-line, and
#             removes it.
#
#   2. textwrap.dedent(): strips the leading space in the string, allowing us
#                         to write more readable multi-line strings.
#
#   3. ESCAPE_RE.sub(): escapes special characters, such as parenthesis,
#                       brackets, and braces, so as to allow us to write more
#                       readable strings.
#
# Note that because of (3), we lose the "regex" part, not being able to use
# regex wildcards, such as "*".
ESCAPE_RE = re.compile(r"([\[\](){}])")


def escape(s):
  return ESCAPE_RE.sub(r"\\\1", textwrap.dedent(s[1:]))


class TestDynamicShapeDetector(test_utils.XlaTestCase):

  def _run_and_compare(self, f, args=None, max_different_graphs=None):
    """Run f and its torch_xla.compile wrapped version, comparing the equality
    of their results.

    If no optf is provided, we create a new one by wrapping it with
    torch_xla.compile ourselves.
    """
    optf = torch_xla.compile(f, max_different_graphs=max_different_graphs)
    args = args or []

    out = f(*args)
    optout = optf(*args)

    self.assertEqual(out, optout)

  def test_single(self):
    # Test: trace a function once, when only one graph is allowed.

    def foo(x):
      return x + x

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(foo, args=(inp,), max_different_graphs=1)

  def test_many_graphs(self):
    # Test: multiple graphs of a function.
    #
    # Steps 0~2 and 5: create new graphs.
    # Steps 3 and 4: ensure we have already traced these paths.

    def foo(x, step):
      r0 = x + x + x
      r = r0 + x
      if step in (0, 3):
        return r + x
      if step == (1, 4):
        return r * 2
      if step == 2:
        return r * 4
      return r0

    inp = torch.rand(10, device=torch_xla.device())

    for i in range(6):
      self._run_and_compare(foo, args=(inp, i), max_different_graphs=4)

  def test_graph_limit_exceeded_different_input_shape(self):
    # Test: catch graph limit exceeded error when running the function with a
    # function with different shape.

    max_different_graphs = 1

    def foo(x):
      return x + x

    inp1 = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(
        foo, args=(inp1,), max_different_graphs=max_different_graphs)

    expected_error_msg = escape(r"""
        Maximum number of different graphs allowed per function exceeded: 1
        Got: [] aten::add, xla_shape=f32[5]{0}, dynamic_dims: ()
        Expected: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
    """)

    with self.assertRaisesRegex(RuntimeError, expected_error_msg):
      inp2 = torch.rand(5, device=torch_xla.device())
      self._run_and_compare(
          foo, args=(inp2,), max_different_graphs=max_different_graphs)

  def test_graph_limit_exceeded_common_sequence_mismatch(self):
    # Test: catch graph limit exceeded error when the common sequence (i.e. compressed
    # path) of the trie node mismatches.
    #
    # Step 0: creates a graph with one node containing the add operation
    #
    # Step 1: tries to create 2 child nodes with:
    # (i) add operation (previous graph); and
    # (ii) mul operation.
    # However, it fails since we have reached the limit.

    max_different_graphs = 1

    def foo(x, step):
      if step == 0:
        return x + x
      else:
        return x * 5

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(
        foo, args=(inp, 0), max_different_graphs=max_different_graphs)

    expected_error_msg = escape(r"""
        Maximum number of different graphs allowed per function exceeded: 1
        Got: [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
        Expected: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
    """)

    with self.assertRaisesRegex(RuntimeError, expected_error_msg):
      self._run_and_compare(
          foo, args=(inp, 2), max_different_graphs=max_different_graphs)

  def test_graph_limit_exceeded_children_mismatch(self):
    # Test: catch graph limit exceeded error when the expected child of the trie
    # node mismatches.
    #
    # Step 0: creates a graph with one node containing 3 operations, the last
    # being a mul operation.
    #
    # Step 1: creates another graph by splitting the node, creating 2 other child
    # nodes containing the different operations in the end:
    # (i) mul operation; and
    # (ii) add operation.
    #
    # Step 2: tries to create a 3rd child node: div operation. However, we can't
    # do it, since we have reached the limit.

    max_different_graphs = 2

    def foo(x, step):
      r = x + x
      if step == 0:
        return r * 2
      if step == 1:
        return r + x
      return r / 3

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(
        foo, args=(inp, 0), max_different_graphs=max_different_graphs)
    self._run_and_compare(
        foo, args=(inp, 1), max_different_graphs=max_different_graphs)

    expected_error_msg = escape(r"""
        Maximum number of different graphs allowed per function exceeded: 2
        Got: [] aten::div, xla_shape=f32[10]{0}, dynamic_dims: ()
        Expected either of:
          - [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
          - [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
    """)

    with self.assertRaisesRegex(RuntimeError, expected_error_msg):
      self._run_and_compare(
          foo, args=(inp, 2), max_different_graphs=max_different_graphs)

  def test_graph_limit_exceeded_common_sequence_early_stop(self):
    # Test: catch graph limit exceeded error when the graph ends unexpectedly in
    # the common sequence.
    #
    # Step 0: creates a graph with one node containing 3 operations.
    #
    # Step 1: at the end of this graph, it tries to create a new node containing
    # the remaining operations of the previous graph, i.e. mul operation. However,
    # it fails because we have reached the limit.

    max_different_graphs = 1

    def foo(x, mul=False):
      r = x + x
      if mul:
        return r * 10
      else:
        return r

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(
        foo, args=(inp, True), max_different_graphs=max_different_graphs)

    expected_error_msg = escape(r"""
        Maximum number of different graphs allowed per function exceeded: 1
        Reached the end of the function at: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
        Expected: [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
    """)

    with self.assertRaisesRegex(RuntimeError, expected_error_msg):
      self._run_and_compare(
          foo, args=(inp, False), max_different_graphs=max_different_graphs)

  def test_graph_limit_exceeded_children_early_stop(self):
    # Test: catch graph limit exceeded error when the graph ends unexpectedly at
    # a fork point (i.e. next operation would jump to anothe trie node).
    #
    # Step 0: creates a graph with one node containing 3 operations.
    #
    # Step 1: splits the node, creating 2 child nodes containing:
    # (i) the differring operations from the last graph, i.e. mul operation
    # (ii) the current last operation, i.e. add operation
    #
    # Step 3: at the end of this graph, it tries to turn the current trie node
    # into a new graph. However, it fails since we have reached the limit.

    max_different_graphs = 2

    def foo(x, step):
      r = x + x
      if step == 0:
        return r * 2
      if step == 1:
        return r + x
      return r

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(
        foo, args=(inp, 0), max_different_graphs=max_different_graphs)
    self._run_and_compare(
        foo, args=(inp, 1), max_different_graphs=max_different_graphs)

    expected_error_msg = escape(r"""
        Maximum number of different graphs allowed per function exceeded: 2
        Reached the end of the function at: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
        Expected either of:
          - [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
          - [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
    """)

    with self.assertRaisesRegex(RuntimeError, expected_error_msg):
      self._run_and_compare(
          foo, args=(inp, 2), max_different_graphs=max_different_graphs)


if __name__ == "__main__":
  unittest.main()
