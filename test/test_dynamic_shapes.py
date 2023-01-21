import os
import sys
import unittest
import torch, torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()


class TestDynamicShapes(unittest.TestCase):

  @unittest.skip("fails with functionalization")
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
    self.assertEqual(t6_cpu.shape[0], 2)  # 10 instead of 2

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

    # size_clone should be called as part of decomposition from
    # the python dispatcher.
    self.assertGreater(met.counter_value("xla::size_clone"), 0)

  def test_simple_expand_add_dimension(self):
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    t1[3][1] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    t3 = torch.ones(1, device=dev)

    t4 = t3.expand(t2.shape[0], t2.shape[0])
    self.assertIsInstance(t4.shape[0], torch.SymInt)
    self.assertEqual(str(t4.shape[0]), '<=10')
    self.assertEqual(t4.shape[0], 2)
    self.assertIsInstance(t4.shape[1], torch.SymInt)
    self.assertEqual(str(t4.shape[1]), '<=10')
    self.assertEqual(t4.shape[1], 2)

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

  def test_sizeGe(self):
    met.clear_all()

    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    # Create a SizeAdd IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] >= t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_ge"), 0)
    # Exercises SizeGe::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 0)

  def test_sizeLt(self):
    met.clear_all()

    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    # Create a SizeAdd IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] < t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_lt"), 0)
    # Exercises SizeLt::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 1)

  def get_dynamic_tensor(self):
    a1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    a2 = torch.nonzero(a1)
    print('a2=', a2)
    return a2

  def test_empty_symint(self):
    # t1.shape= torch.Size([<=6, 2]) with real size [3, 2]
    t1 = self.get_dynamic_tensor()
    print('t1=', t1)
    self.assertIsInstance(t1.shape[0], torch.SymInt)
    t2 = torch.empty(t1.shape, dtype=torch.int32, device=dev)
    self.assertIsInstance(t2.shape[0], torch.SymInt)
    self.assertEqual(str(t2.shape[0]), '<=6')
    self.assertEqual(t2.shape[0], 3)
    self.assertIsInstance(t2.shape[1], int)
    self.assertEqual(t2.shape[1], 2)
    



if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main()
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
