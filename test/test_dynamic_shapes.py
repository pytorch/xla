import os
import sys
import unittest
import torch, torch_xla
import torch_xla.core.xla_model as xm

dev = xm.xla_device()


class TestDynamicShapes(unittest.TestCase):

  def test_simple_expand(self):
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    t1[3][1] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    t5 = torch.ones(1, device=dev)
    t6 = t5.expand(t2.size(0))
    self.assertIn('<=10', torch_xla._XLAC._get_xla_tensors_text([t6]))
    t6_cpu = t6.cpu()
    self.assertEqual(t6_cpu.shape[0], 2)

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
    a1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    a2 = torch.nonzero(a1)
    self.assertTrue(a2.shape[0] == 3)
    a3 = a2.shape[0] + 3  # tests wrap
    self.assertIsInstance(a3, torch.SymInt)

  def test_sizeAdd(self):
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
  
  def get_dynamic_tensor(self):
    a1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    a2 = torch.nonzero(a1)
    print('a2=', a2)
    return a2

  def test_xla_add(self):
    # t1.shape= torch.Size([<=6, 2])
    t1 = self.get_dynamic_tensor()
    t2 = self.get_dynamic_tensor()
    self.assertIsInstance(t1.shape[0], torch.SymInt)
    self.assertIsInstance(t2.shape[0], torch.SymInt)
    t3 = t1 + t2
    self.assertIsInstance(t3.shape[0], torch.SymInt)

  def test_xla_fill_(self):
    # t1.shape= torch.Size([<=6, 2])
    t1 = self.get_dynamic_tensor()
    print('t1=', t1)
    self.assertIsInstance(t1.shape[0], torch.SymInt)
    t2 = t1.fill_(10)
    self.assertIsInstance(t2.shape[0], torch.SymInt)

  def test_xla_mm(self):
    # t1.shape= torch.Size([<=6, 2])
    t1 = self.get_dynamic_tensor() 
    t2 = torch.ones((2, 2), device=dev)
    t3= torch.mm(t1, t2)
    self.assertIsInstance(t3.shape[0], torch.SymInt)

  def test_xla_unsqueeze(self):
    # t1.shape= torch.Size([<=6, 2])
    t1 = self.get_dynamic_tensor() 
    t2 = t1.unsqueeze(dim=0)
    self.assertIsInstance(t2.shape[0], int)
    self.assertIsInstance(t2.shape[1], torch.SymInt)


  def test_xla_view_symint(self):
    # t1.shape= torch.Size([<=6, 2])
    t1 = self.get_dynamic_tensor()  
    # TODO: xiowei continue from here.
    
  



if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
