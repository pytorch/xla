import torch, torch_xla
import torch_xla.core.xla_model as xm
import unittest

dev = xm.xla_device()


class TestDynamicShapes(unittest.TestCase):

  def test_wrap(self):
    a1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    a2 = torch.nonzero(a1)
    self.assertTrue(a2.shape[0] == 3)
    a3 = a2.shape[0] + 3  # tests wrap
    self.assertIsInstance(a3, torch.SymInt)

  def test_sizeAdd(self):
    from inspect import currentframe
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    print("xw32", currentframe().f_lineno)
    t1[3][0] = 1
    print("xw32", currentframe().f_lineno)
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    print("xw32", currentframe().f_lineno)
    t4 = t2.shape[0] + t2.shape[1]
    print("xw32", currentframe().f_lineno)
    print(t4)
    print("xw32", currentframe().f_lineno)
    print(t4.cpu())
    print("xw32", currentframe().f_lineno)
    xm.mark_step()
    print('done')    


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
