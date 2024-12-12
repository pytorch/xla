import os
import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm


class InplaceUpdateTest(unittest.TestCase):

  def test_aten_op_after_full_update(self):
    device = xm.xla_device()
    t = torch.ones(2, 1, device=device)
    w = torch.ones(1, 2, device=device)
    t.zero_()
    y = torch.matmul(t, w)
    expected = torch.zeros(2, 2, device=device)
    xm.mark_step()
    self.assertTrue(torch.all(torch.eq(y, expected)))

  def test_aten_op_after_partial_update(self):
    device = xm.xla_device()
    t = torch.ones(2, 1, device=device)
    w = torch.ones(1, 2, device=device)
    t[0][0] = 0
    y = torch.matmul(t, w)
    expected = torch.tensor([[0, 0], [1, 1]], device=device)
    xm.mark_step()
    self.assertTrue(torch.all(torch.eq(y, expected)))

  def test_non_aten_op_after_full_update(self):
    device = xm.xla_device()
    t = torch.ones(2, 1, device=device)
    w = torch.ones(1, 2, device=device)
    t.zero_()
    y = torch_xla._XLAC._xla_dot_general(t, w, (([1], [0]), ()))
    expected = torch.zeros(2, 2, device=device)
    xm.mark_step()
    self.assertTrue(torch.all(torch.eq(y, expected)))

  def test_non_aten_op_after_partial_update(self):
    device = xm.xla_device()
    t = torch.ones(2, 1, device=device)
    w = torch.ones(1, 2, device=device)
    t[0][0] = 0
    y = torch_xla._XLAC._xla_dot_general(t, w, (([1], [0]), ()))
    expected = torch.tensor([[0, 0], [1, 1]], device=device)
    xm.mark_step()
    self.assertTrue(torch.all(torch.eq(y, expected)))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
