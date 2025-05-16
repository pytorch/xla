"""Tests for asynchronous closures."""
from threading import Event
from time import sleep

import unittest

import torch_xla
import torch_xla.core.xla_model as xm


class AsyncClosuresTest(unittest.TestCase):

  def test_synchronous(self):
    flag = Event()
    assert not flag.is_set()

    def closure():
      sleep(1)
      assert not flag.is_set()
      flag.set()

    xm.add_step_closure(closure)
    torch_xla.sync()

    # should not get to this part before closure is finished running
    assert flag.is_set()

  def test_asynchronous(self):
    flag = Event()
    assert not flag.is_set()

    def closure():
      sleep(1)
      assert flag.is_set()

    xm.add_step_closure(closure, run_async=True)
    torch_xla.sync()

    # should get to this part and complete before closure is finished running
    assert not flag.is_set()
    flag.set()

  def test_synchronous_exception(self):
    flag = Event()
    assert not flag.is_set()

    try:

      def closure():
        flag.set()
        raise RuntimeError("Simulating exception in closure")

      xm.add_step_closure(closure)
      torch_xla.sync()

      assert False  # Should not reach here
    except RuntimeError as e:
      assert flag.is_set(), "Should have caught exception from closure"

  def test_asynchronous_exception(self):
    flag = Event()
    assert not flag.is_set()

    def closure1():
      flag.set()
      raise RuntimeError("Simulating exception in closure1")

    xm.add_step_closure(closure1, run_async=True)
    torch_xla.sync()

    flag.wait(timeout=5)

    try:

      def closure2():  # Should never execute
        flag.clear()

      xm.add_step_closure(closure2, run_async=True)
      torch_xla.sync()

      assert False  # Should not reach here
    except RuntimeError as e:
      # Should have caught exception from closure1
      pass

    assert flag.is_set()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
