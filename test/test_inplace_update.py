import io
import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from test_utils import temporary_env


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

  def test_xm_save(self):
    with temporary_env(
        XLA_DISABLE_FUNCTIONALIZATION="0", XLA_ENABLE_PARAM_ALIASING="0"):
      xla_device = xm.xla_device()
      t1 = torch.tensor([1], device=xla_device)
      t2 = t1.detach()
      xm.mark_step()

      t2.add_(t2)
      xm.mark_step()

      # mark_step() causes t1 and t2 to be out of sync on the XLA side.

      fobj = io.BytesIO()
      xm.save({'t1': t1}, fobj)
      fobj.seek(0)
      saved = torch.load(fobj)

      self.assertEqual(t1.item(), saved['t1'].item())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
