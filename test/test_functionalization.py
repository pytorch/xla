import io
import os

os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["XLA_ENABLE_PARAM_ALIASING"] = "0"

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest


class TestFunctionalization(unittest.TestCase):

  def test_xm_save(self):
    """
    Test that xm.save() does torch._functionalize_sync()
    """
    xla_device = xm.xla_device()
    t1 = torch.tensor([1], device=xla_device)
    t2 = t1.detach()
    xm.mark_step()

    t2.add_(t2)
    xm.mark_step()

    # mark_step() causes t1 and t2 to be out of sync on the XLA side.
    # _functionalize_sync() is needed to get them back in sync.

    fobj = io.BytesIO()
    xm.save({'t1': t1}, fobj)
    fobj.seek(0)
    saved = torch.load(fobj)

    self.assertEqual(t1.item(), saved['t1'].item())


if __name__ == "__main__":
  unittest.main()
