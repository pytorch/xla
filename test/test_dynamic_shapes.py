import os
import unittest
import torch, torch_xla
import torch_xla.core.xla_model as xm

dev = xm.xla_device()


class TestDynamicShapes(unittest.TestCase):

  def test_simple_expand(self):
    self.assertNotEqual(os.environ['XLA_EXPERIMENTAL'], '')
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    t1[3][1] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    t5 = torch.ones(1, device=dev)
    t6 = t5.expand(t2.size(0))
    print(torch_xla._XLAC._get_xla_tensors_text([t6]))
    t6_cpu = t6.cpu()
    self.assertEqual(t6_cpu.shape[0], 2)
    print(torch_xla._XLAC._get_xla_tensors_text([t6]))

  def test_simple_expand_on_2d_tensor(self):
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    t1[3][1] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    t3 = torch.ones(1, size2, device=dev)

    # varargs
    t4 = t3.expand(t2.shape[0], t2.shape[1])
    self.assertEqual(t4.shape[0], 2)
    self.assertEqual(t4.shape[1], size2)

    # shape list
    t4 = t3.expand((t2.shape[0], t2.shape[1]))
    self.assertEqual(t4.shape[0], 2)
    self.assertEqual(t4.shape[1], size2)

    # mixed python symints and ints
    t4 = t3.expand(t2.shape[0], size2)
    self.assertEqual(t4.shape[0], 2)
    self.assertEqual(t4.shape[1], size2)

    # mixed python symints and ints in a list
    t4 = t3.expand((t2.shape[0], size2))
    self.assertEqual(t4.shape[0], 2)
    self.assertEqual(t4.shape[1], size2)

  def test_wrap(self):
    self.assertNotEqual(os.environ['XLA_EXPERIMENTAL'], '')
    a1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    a2 = torch.nonzero(a1)
    self.assertTrue(a2.shape[0] == 3)
    a3 = a2.shape[0] + 3  # tests wrap
    self.assertIsInstance(a3, torch.SymInt)

  def test_sizeAdd(self):
    self.assertNotEqual(os.environ['XLA_EXPERIMENTAL'], '')
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    # Create a SizeAdd IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] + t2.shape[1]
    # Exercises SizeAdd::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 3)
    # Exercise SizeAdd::getStaticValue.
    self.assertEqual(str(dyn_size), '<=12')
    t3 = torch.ones(1, device=dev)
    # Exercise SizeAdd::Lower.
    t4 = t3.expand(dyn_size)
    self.assertEqual(t4.size(0), 3)
    print(torch_xla._XLAC._get_xla_tensors_text([t4]))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
