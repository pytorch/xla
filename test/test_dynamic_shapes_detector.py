import torch
import torch_xla
import test_utils
import unittest


def set_max_allowed_traces(traces):
  torch_xla._XLAC._dynamic_shape_detector_set_max_allowed_traces(traces)


class TestDynamicShapeDetector(test_utils.XlaTestCase):

  def _run_and_compare(self, f, optf=None, args=None):
    """Run f and its torch_xla.compile wrapped version, comparing the equality
    of their results.

    If no optf is provided, we create a new one by wrapping it with
    torch_xla.compile ourselves.
    """
    optf = optf or torch_xla.compile(f, detect_dynamic_shape=True)
    args = args or []

    out = f(*args)
    optout = optf(*args)

    self.assertEqual(out, optout)

  def test_single(self):
    # Test: trace a function once, when only one trace is allowed.

    set_max_allowed_traces(1)

    def foo(x):
      return x + x

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(foo, args=(inp,))

  def test_many_traces(self):
    # Test: multiple traces of a function.
    #
    # Steps 0~2 and 5: create new traces.
    # Steps 3 and 4: ensure we have already traced these paths.

    set_max_allowed_traces(4)

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

    optfoo = torch_xla.compile(foo, detect_dynamic_shape=True)
    inp = torch.rand(10, device=torch_xla.device())

    for i in range(6):
      self._run_and_compare(foo, args=(inp, i))

  def test_trace_limit_exceeded_different_input_shape(self):
    # Test: catch trace limit exceeded error when running the function with a
    # function with different shape.

    set_max_allowed_traces(1)

    def foo(x):
      return x + x

    optfoo = torch_xla.compile(foo, detect_dynamic_shape=True)

    inp1 = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(foo, optfoo, args=(inp1,))

    msg = """\
torch_xla/csrc/dynamic_shape_detector.cpp:47 : Maximum number of different traces allowed per function exceeded: 1
Got: [] aten::expand, xla_shape=f32[10]{0}, dynamic_dims: (), size=(10)
Expected: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()"""

    with self.assertRaises(RuntimeError, msg=msg):
      inp2 = torch.rand(5, device=torch_xla.device())
      self._run_and_compare(foo, optfoo, args=(inp2,))

  def test_trace_limit_exceeded_common_sequence_mismatch(self):
    # Test: catch trace limit exceeded error when the common sequence (i.e. compressed
    # path) of the trie node mismatches.
    #
    # Step 0: creates a trace with one node containing the add operation
    #
    # Step 1: tries to create 2 child nodes with:
    # (i) add operation (previous trace); and
    # (ii) mul operation.
    # However, it fails since we have reached the limit.

    set_max_allowed_traces(1)

    def foo(x, step):
      if step == 0:
        return x + x
      else:
        return x * 5

    optfoo = torch_xla.compile(foo, detect_dynamic_shape=True)

    inp = torch.rand(10, device=torch_xla.device())
    self._run_and_compare(foo, optfoo, args=(inp, 0))

    msg = """\
torch_xla/csrc/dynamic_shape_detector.cpp:47 : Maximum number of different traces allowed per function exceeded: 1
Got: [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
Expected: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()"""

    with self.assertRaises(RuntimeError, msg=msg):
      self._run_and_compare(foo, optfoo, args=(inp, 2))

  def test_trace_limit_exceeded_children_mismatch(self):
    # Test: catch trace limit exceeded error when the expected child of the trie
    # node mismatches.
    #
    # Step 0: creates a trace with one node containing 3 operations, the last
    # being a mul operation.
    #
    # Step 1: creates another trace by splitting the node, creating 2 other child
    # nodes containing the different operations in the end:
    # (i) mul operation; and
    # (ii) add operation.
    #
    # Step 2: tries to create a 3rd child node: div operation. However, we can't
    # do it, since we have reached the limit.

    set_max_allowed_traces(2)

    def foo(x, step):
      r = x + x
      if step == 0:
        return r * 2
      if step == 1:
        return r + x
      return r / 3

    optfoo = torch_xla.compile(foo, detect_dynamic_shape=True)
    inp = torch.rand(10, device=torch_xla.device())

    self._run_and_compare(foo, optfoo, args=(inp, 0))
    self._run_and_compare(foo, optfoo, args=(inp, 1))

    msg = """\
torch_xla/csrc/dynamic_shape_detector.cpp:47 : Maximum number of different traces allowed per function exceeded: 2
Got: [] aten::expand, xla_shape=f32[10]{0}, dynamic_dims: (), size=(10)
Expected either of:
  - [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
  - [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()"""

    with self.assertRaises(RuntimeError, msg=msg):
      self._run_and_compare(foo, optfoo, args=(inp, 2))

  def test_trace_limit_exceeded_common_sequence_early_stop(self):
    # Test: catch trace limit exceeded error when the trace ends unexpectedly in
    # the common sequence.
    #
    # Step 0: creates a trace with one node containing 3 operations.
    #
    # Step 1: at the end of this trace, it tries to create a new node containing
    # the remaining operations of the previous trace, i.e. mul operation. However,
    # it fails because we have reached the limit.

    set_max_allowed_traces(1)

    def foo(x, mul=False):
      r = x + x
      if mul:
        return r * 10
      else:
        return r

    optfoo = torch_xla.compile(foo, detect_dynamic_shape=True)
    inp = torch.rand(10, device=torch_xla.device())

    self._run_and_compare(foo, optfoo, args=(inp, True))

    msg = """\
torch_xla/csrc/dynamic_shape_detector.cpp:47 : Maximum number of different traces allowed per function exceeded: 1
Reached the end of the function at: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
Expected: [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()"""

    with self.assertRaises(RuntimeError, msg=msg):
      self._run_and_compare(foo, optfoo, args=(inp, False))

  def test_trace_limit_exceeded_children_early_stop(self):
    # Test: catch trace limit exceeded error when the trace ends unexpectedly at
    # a fork point (i.e. next operation would jump to anothe trie node).
    #
    # Step 0: creates a trace with one node containing 3 operations.
    #
    # Step 1: splits the node, creating 2 child nodes containing:
    # (i) the differring operations from the last trace, i.e. mul operation
    # (ii) the current last operation, i.e. add operation
    #
    # Step 3: at the end of this trace, it tries to turn the current trie node
    # into a new trace. However, it fails since we have reached the limit.

    set_max_allowed_traces(2)

    def foo(x, step):
      r = x + x
      if step == 0:
        return r * 2
      if step == 1:
        return r + x
      return r

    optfoo = torch_xla.compile(foo, detect_dynamic_shape=True)
    inp = torch.rand(10, device=torch_xla.device())

    self._run_and_compare(foo, optfoo, args=(inp, 0))
    self._run_and_compare(foo, optfoo, args=(inp, 1))

    msg = """\
torch_xla/csrc/dynamic_shape_detector.cpp:47 : Maximum number of different traces allowed per function exceeded: 2
Reached the end of the function at: [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()
Expected either of:
  - [] aten::mul, xla_shape=f32[10]{0}, dynamic_dims: ()
  - [] aten::add, xla_shape=f32[10]{0}, dynamic_dims: ()"""

    with self.assertRaises(RuntimeError, msg=msg):
      self._run_and_compare(foo, optfoo, args=(inp, 2))


if __name__ == "__main__":
  unittest.main()
