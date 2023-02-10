import os
import sys
import unittest
import torch, torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import test_utils

pd = torch._C._EnablePythonDispatcher()
dev = xm.xla_device()


class TestDynamicShapes(test_utils.XlaTestCase):

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

  def test_sizeSub(self):
    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[0][0] = 1
    t1[1][0] = 1
    t1[2][0] = 1
    # t2 has size [<=10, 2] with dynamic size=[3, 2]
    t2 = torch.nonzero(t1)
    dyn_size = t2.shape[0] - t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_sub"), 0)
    # Exercises SizeSub::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 1)
    # Exercise SizeSub::getStaticValue.
    self.assertEqual(str(dyn_size), '<=8')

    t3 = torch.ones(1, device=dev)
    # Exercise SizeSub::Lower.
    t4 = t3.expand(dyn_size)
    self.assertEqual(t4.size(0), 1)

  def get_dynamic_tensor(self):
    a1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    a2 = torch.nonzero(a1)
    return a2

  def test_empty_symint(self):
    # t1.shape= torch.Size([<=6, 2]) with real size [3, 2]
    t1 = self.get_dynamic_tensor()
    # Don't print t1 otherwise it would cause the test to crash.
    self.assertIsInstance(t1.shape[0], torch.SymInt)
    t2 = torch.empty(t1.shape, dtype=torch.int32, device=dev)
    self.assertIsInstance(t2.shape[0], torch.SymInt)
    self.assertEqual(str(t2.shape[0]), '<=6')
    self.assertEqual(t2.shape[0], 3)
    self.assertIsInstance(t2.shape[1], int)
    self.assertEqual(t2.shape[1], 2)

  def test_nonzero_shape(self):
    x = torch.tensor((0, 1, 2, 0, 3, 4), device=xm.xla_device())
    x_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(
        torch.nonzero(x, as_tuple=False), 0)
    self.assertEqual(x_dim0_shape.item(), 4)

  def test_masked_select_shape(self):
    x = torch.tensor((0, 1, 2, 0, 3, 4), device=xm.xla_device())
    mask = x.ge(2)
    x_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(
        torch.masked_select(x, mask), 0)
    self.assertEqual(x_dim0_shape.item(), 3)

  def test_nonzero_cast(self):
    t1 = torch.ones(5, 2, device=xm.xla_device())
    # Result of the nonzero should be the index type. Currently
    # index type is s64 on cpu and gpu, but s32 on TPU. We should be
    # able to cast it to any other type without error.
    t2 = torch.nonzero(t1.int()).float()
    xm.mark_step()

  def test_expand_symint_correctness(self):
    dev = xm.xla_device()
    size1 = 5
    size2 = 2
    t1 = torch.ones([size1, size2])
    expand_out_aten = t1.expand(2, size1, size2)

    t2 = torch.zeros([size1, size2], device=dev)
    t2[3][0] = 1
    t2[3][1] = 1
    # t2 has size [<=10, 2]
    t3 = torch.nonzero(t2)
    t4 = torch.ones([size1, size2], device=dev)
    expand_out_xla = t4.expand(t3.shape[0], size1, size2)
    self.assertEqual(t3.shape[0], 2)
    self.assertEqual(expand_out_aten.cpu(), expand_out_xla.cpu())


if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main()
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
