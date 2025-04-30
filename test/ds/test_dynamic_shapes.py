import os
import sys
import unittest
import torch, torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

sys.path.insert(1, os.path.join(sys.path[0], '..'))
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

  def test_t_copy(self):
    t1 = torch.tensor([[1, 0, 0, 5, 0, 6], [1, 3, 2, 0, 0, 1]], device=dev)
    t2 = torch.nonzero(t1)
    # t2.shape=torch.Size([<=12, 2]) with real size [7, 2]
    self.assertEqual(str(t2.shape[0]), '<=12')
    self.assertEqual(str(t2.shape[1]), '2')

    t2_t = torch.t(t2)

    self.assertIsInstance(t2_t.shape[0], int)
    self.assertIsInstance(t2_t.shape[1], torch.SymInt)
    self.assertEqual(str(t2_t.shape[0]), '2')
    self.assertEqual(str(t2_t.shape[1]), '<=12')
    self.assertEqual(t2_t.shape[0], 2)
    self.assertEqual(t2_t.shape[1], 7)

  def test_nonzero_shape(self):
    x = torch.tensor((0, 1, 2, 0, 3, 4), device=xm.xla_device())
    x_dim0_shape = torch_xla._XLAC._get_xla_tensor_dimension_size(
        torch.nonzero(x, as_tuple=False), 0)
    self.assertEqual(x_dim0_shape.item(), 4)

  def test_nonzero_correctness(self):
    t1 = torch.tensor([[1, 0, 0, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t1_aten = t1.cpu()
    t2_aten = torch.nonzero(t1_aten)
    self.assertEqual(t2.cpu(), t2_aten)

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
    torch_xla.sync()

  def test_expand_symint_correctness(self):
    dev = xm.xla_device()
    size1 = 5
    size2 = 2
    t1 = torch.ones([size1, size2])
    expand_out_aten = t1.expand(2, size1, size2)

    t2 = torch.zeros([size1, size2], device=dev)
    t2[3][0] = 1
    t2[3][1] = 1
    # t3 has size [<=10, 2]
    t3 = torch.nonzero(t2)
    t4 = torch.ones([size1, size2], device=dev)
    expand_out_xla = t4.expand(t3.shape[0], size1, size2)
    self.assertEqual(t3.shape[0], 2)
    self.assertEqual(expand_out_aten.cpu(), expand_out_xla.cpu())

  def test_unsqueeze_copy_dynamism(self):
    t1 = torch.tensor([[1, 0, 0, 5, 0, 6], [1, 3, 2, 0, 0, 1]], device=dev)
    t2 = torch.nonzero(t1)
    # t2.shape=torch.Size([<=12, 2]) with real size [7, 2]

    t2_unsqueeze = torch.unsqueeze(t2, 0)

    self.assertEqual(len(t2_unsqueeze.size()), 3)
    self.assertIsInstance(t2_unsqueeze.shape[0], int)
    self.assertIsInstance(t2_unsqueeze.shape[1], torch.SymInt)
    self.assertIsInstance(t2_unsqueeze.shape[2], int)
    self.assertEqual(str(t2_unsqueeze.shape[0]), '1')
    self.assertEqual(str(t2_unsqueeze.shape[1]), '<=12')
    self.assertEqual(str(t2_unsqueeze.shape[2]), '2')
    self.assertEqual(t2_unsqueeze.shape[0], 1)
    self.assertEqual(t2_unsqueeze.shape[1], 7)
    self.assertEqual(t2_unsqueeze.shape[2], 2)

    # test correctness
    t3 = torch.tensor([[1, 0, 0, 5, 0, 6], [1, 3, 2, 0, 0, 1]])
    t4 = torch.nonzero(t3)
    t4_unsqueeze = torch.unsqueeze(t4, 0)
    self.assertEqual(t2_unsqueeze.cpu(), t4_unsqueeze.cpu())

  def test_view_copy_symint_with_static_input_dyn_input_shape(self):
    # If the input tensor and shape are “statically” incompatible, a compilation error is raised.
    t1 = torch.tensor([1, 0, 3, 5, 0, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [4, 1]
    # t2 = [[0], [2], [3], [5]]
    t2 = torch.nonzero(t1)
    t3 = torch.randint(10, (2, 2), device=dev)
    self.assertRaises(RuntimeError, lambda: t3.view(t2.shape[0]))

    # If their “dynamic” values are incompatible, a RuntimeError is raised.
    t4 = torch.randint(10, (2, 3), device=dev)
    self.assertRaises(RuntimeError, lambda: t4.view(t2.shape[0]))

    # verify if dynamism is propagated correctly.
    t5 = torch.tensor([1, 1, 3, 5, 1, 6], device=dev)
    t6 = torch.nonzero(t5)
    t7 = torch.ones((2, 3), device=dev)
    # t6.shape=torch.Size([<=6, 1]) with real size [6, 1]
    # t6 = [[0], [1], [2], [3], [4], [5]]
    t8 = t7.view(t6.shape[0])
    self.assertIsInstance(t8.shape[0], torch.SymInt)
    self.assertEqual(str(t8.shape[0]), '<=6')
    self.assertEqual(t8.shape[0], 6)

    # verify correctness.
    t5_aten = torch.tensor([1, 1, 3, 5, 1, 6])
    t6_aten = torch.nonzero(t5_aten)
    t7_aten = torch.ones((2, 3))
    t8_aten = t7_aten.view(t6_aten.shape[0])
    self.assertEqual(t8.cpu(), t8_aten.cpu())

  def test_view_copy_symint_with_static_input_dyn_input_shape2(self):
    # If the input tensor and shape are “statically” incompatible, a compilation error is raised.
    t1 = torch.tensor([[1, 0, 3]], device=dev)
    # t2.shape=torch.Size([<=3, 2]) with real size [2, 2]
    # t2 = [[0, 0], [0, 2]]
    t2 = torch.nonzero(t1)
    t3 = torch.ones((2, 4), device=dev)
    # Should fail in pytorch utils.infer_size
    self.assertRaises(RuntimeError, lambda: t3.view(t2.shape))

    # If their “dynamic” values are incompatible, a RuntimeError is raised.
    t4 = torch.ones((2, 3), device=dev)
    # Also fails in pytorch utils.infer_size
    self.assertRaises(RuntimeError, lambda: t4.view(t2.shape))

    # verify if dynamism is propagated correctly.
    t5 = torch.tensor([[1, 1, 3]], device=dev)
    t6 = torch.nonzero(t5)
    # t6.shape=[<=3, 2] with real size [3, 2]
    t7 = torch.ones((2, 3), device=dev)
    t8 = t7.view(t6.shape)
    self.assertIsInstance(t8.shape[0], torch.SymInt)
    self.assertEqual(str(t8.shape[0]), '<=3')
    self.assertEqual(t8.shape[0], 3)
    self.assertIsInstance(t8.shape[1], int)
    self.assertEqual(str(t8.shape[1]), '2')
    self.assertEqual(t8.shape[1], 2)

    # verify correctness.
    t5_aten = torch.tensor([[1, 1, 3]])
    t6_aten = torch.nonzero(t5_aten)
    t7_aten = torch.ones((2, 3))
    t8_aten = t7_aten.view(t6_aten.shape)
    self.assertEqual(t8.cpu(), t8_aten.cpu())

  def test_view_copy_symint_with_dyn_input_static_input_shape(self):
    # If the input tensor is dynamic and input shape is static,
    # it should fail because we will not likely have this case
    # in reality so we don't support this feature.
    t1 = torch.tensor([1, 1, 3, 5, 1, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [6, 1]
    t2 = torch.nonzero(t1)
    self.assertRaises(RuntimeError, lambda: t2.view(2, 3))

  def test_view_copy_symint_with_dyn_input_dyn_input_shape(self):
    # If the input tensor and shape are “statically” incompatible, a compilation error is raised.
    t1 = torch.tensor([1, 0, 3, 5, 0, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [4, 1]
    # t2 = [[0], [2], [3], [5]]
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([1, 0, 3, 5, 0, 6, 7], device=dev)
    # t4.shape=torch.Size([<=7, 1]) with real size [5, 1]
    t4 = torch.nonzero(t3)
    self.assertRaises(RuntimeError, lambda: t2.view(t4.shape[0]))

    # If their “dynamic” values are incompatible, a RuntimeError is raised.
    t5 = torch.tensor([1, 2, 3, 4, 5, 6, 0], device=dev)
    # t6.shape=torch.Size([<=7, 1]) with real size [6, 1]
    t6 = torch.nonzero(t5)
    # statically compatible but dynamically incompatible.
    # It will fail in pytorch layer.
    self.assertRaises(RuntimeError, lambda: t6.view(t4.shape[0]))

    # verify if dynamism is propagated correctly.
    t7 = torch.tensor([1, 0, 3, 5, 0, 6, 7], device=dev)
    t8 = torch.nonzero(t7)
    # t8.shape=torch.Size([<=7, 1]) with real size [5, 1]
    t9 = t8.view(t4.shape[0])
    self.assertIsInstance(t9.shape[0], torch.SymInt)
    self.assertEqual(str(t9.shape[0]), '<=7')
    self.assertEqual(t9.shape[0], 5)

    # verify correctness.
    t7_aten = torch.tensor([1, 0, 3, 5, 0, 6, 7])
    t8_aten = torch.nonzero(t7_aten)
    # t8_aten.size=[5, 1]
    t3_aten = torch.tensor([1, 0, 3, 5, 0, 6, 7])
    t4_aten = torch.nonzero(t3_aten)
    # t4_aten.size=[5, 1]
    t9_aten = t8_aten.view(t4_aten.shape[0])
    self.assertEqual(t9.cpu(), t9_aten.cpu())

  def test_add_dyn_with_static_broadcastable(self):
    t1 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([[1, 1]], device=dev)

    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t3.shape=torch.Size([1, 2]) with real size [1, 2]
    t4 = torch.add(t2, t3)
    self.assertIsInstance(t4.shape[0], torch.SymInt)
    self.assertEqual(str(t4.shape[0]), '<=6')
    self.assertEqual(t4.shape[0], 4)
    self.assertIsInstance(t4.shape[1], int)
    self.assertEqual(str(t4.shape[1]), '2')
    self.assertEqual(t4.shape[1], 2)

    # test for correctness
    t1_aten = torch.tensor([[1, 0, 3, 5, 0, 6]])
    t2_aten = torch.nonzero(t1_aten)
    t3_aten = torch.tensor([[1, 1]])
    t4_aten = torch.add(t2_aten, t3_aten)
    self.assertEqual(t4.cpu(), t4_aten.cpu())

  def test_add_dyn_with_static_not_broadcastable(self):
    t1 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([[1, 1], [1, 1]], device=dev)

    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t3.shape=torch.Size([2, 2]) with real size [2, 2]
    self.assertRaises(RuntimeError, lambda: torch.add(t2, t3))
    self.assertRaises(RuntimeError, lambda: torch.add(t3, t2))

  def test_add_two_dynamic_tensors(self):
    t1 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([[1]], device=dev)
    t4 = torch.nonzero(t3)

    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t4.shape=torch.Size([<=1, 2]) with real size [1, 2]
    self.assertRaises(RuntimeError, lambda: torch.add(t2, t4))
    self.assertRaises(RuntimeError, lambda: torch.add(t4, t2))

    # For now, we disallow if both operands have the same upper bound and real size.
    # This is consistent with PyTorch's behavior.
    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t6.shape=torch.Size([<=6, 2]) with real size [4, 2]
    t5 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t6 = torch.nonzero(t5)
    self.assertRaises(RuntimeError, lambda: torch.add(t2, t6))

  def test_sub_dyn_with_static_broadcastable(self):
    t1 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([[1, 1]], device=dev)

    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t3.shape=torch.Size([1, 2]) with real size [1, 2]
    t4 = torch.sub(t2, t3)
    self.assertIsInstance(t4.shape[0], torch.SymInt)
    self.assertEqual(str(t4.shape[0]), '<=6')
    self.assertEqual(t4.shape[0], 4)
    self.assertIsInstance(t4.shape[1], int)
    self.assertEqual(str(t4.shape[1]), '2')
    self.assertEqual(t4.shape[1], 2)

    # test for correctness
    t1_aten = torch.tensor([[1, 0, 3, 5, 0, 6]])
    t2_aten = torch.nonzero(t1_aten)
    t3_aten = torch.tensor([[1, 1]])
    t4_aten = torch.sub(t2_aten, t3_aten)
    self.assertEqual(t4.cpu(), t4_aten.cpu())

  def test_sub_dyn_with_static_not_broadcastable(self):
    t1 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([[1, 1], [1, 1]], device=dev)

    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t3.shape=torch.Size([2, 2]) with real size [2, 2]
    self.assertRaises(RuntimeError, lambda: torch.sub(t2, t3))
    self.assertRaises(RuntimeError, lambda: torch.sub(t3, t2))

  def test_sub_two_dynamic_tensors(self):
    t1 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t2 = torch.nonzero(t1)
    t3 = torch.tensor([[1]], device=dev)
    t4 = torch.nonzero(t3)

    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t4.shape=torch.Size([<=1, 2]) with real size [1, 2]
    self.assertRaises(RuntimeError, lambda: torch.sub(t2, t4))
    self.assertRaises(RuntimeError, lambda: torch.sub(t4, t2))

    # For now, we disallow if both operands have the same upper bound and real size.
    # This is consistent with PyTorch's behavior.
    # t2.shape=torch.Size([<=6, 2]) with real size [4, 2]
    # t6.shape=torch.Size([<=6, 2]) with real size [4, 2]
    t5 = torch.tensor([[1, 0, 3, 5, 0, 6]], device=dev)
    t6 = torch.nonzero(t5)
    self.assertRaises(RuntimeError, lambda: torch.sub(t2, t6))
    self.assertRaises(RuntimeError, lambda: torch.sub(t6, t2))

  def test_clone(self):
    t1 = torch.tensor([1, 0, 3, 5, 0, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [4, 1]
    # t2 = [[0], [2], [3], [5]]
    t2 = torch.nonzero(t1)
    t2_clone = torch.clone(t2)
    self.assertIsInstance(t2_clone.shape[0], torch.SymInt)
    self.assertEqual(str(t2_clone.shape[0]), '<=6')
    self.assertEqual(t2_clone.shape[0], 4)
    self.assertIsInstance(t2_clone.shape[1], int)
    self.assertEqual(str(t2_clone.shape[1]), '1')
    self.assertEqual(t2_clone.shape[1], 1)

    # For correctness
    self.assertEqual(t2.cpu(), t2_clone.cpu())

  def test_xlatensor_memoize_symsizes(self):
    met.clear_all()
    t1 = torch.tensor([1, 0, 3, 5, 0, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [4, 1]
    # t2 = [[0], [2], [3], [5]]
    t2 = torch.nonzero(t1)
    sym_size0 = t2.shape[0]
    sym_size1 = t2.shape[0]
    self.assertEqual(sym_size0, sym_size1)
    self.assertIsNone(met.metric_data('CompileTime'))

  def test_abs(self):
    t1 = torch.tensor([1, 0, 3, 5, 0, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [4, 1]
    # t2 = [[0], [2], [3], [5]]
    t2 = torch.nonzero(t1)
    t3 = torch.abs(t2)
    self.assertIsInstance(t3.shape[0], torch.SymInt)
    self.assertEqual(str(t3.shape[0]), '<=6')
    self.assertEqual(t3.shape[0], 4)
    self.assertIsInstance(t3.shape[1], int)
    self.assertEqual(str(t3.shape[1]), '1')
    self.assertEqual(t3.shape[1], 1)

    # test for correctness
    t1_aten = torch.tensor([1, 0, 3, 5, 0, 6])
    t2_aten = torch.nonzero(t1_aten)
    t3_aten = torch.abs(t2_aten)
    self.assertEqual(t3.cpu(), t3_aten.cpu())

  def test_fill_(self):
    t1 = torch.tensor([1, 0, 3, 5, 0, 6], device=dev)
    # t2.shape=torch.Size([<=6, 1]) with real size [4, 1]
    # t2 = [[0], [2], [3], [5]]
    t2 = torch.nonzero(t1)
    self.assertIsInstance(t2.shape[0], torch.SymInt)
    self.assertIsInstance(t2.shape[1], int)

    t2.fill_(1)
    self.assertIsInstance(t2.shape[0], torch.SymInt)
    self.assertEqual(str(t2.shape[0]), '<=6')
    self.assertEqual(t2.shape[0], 4)
    self.assertIsInstance(t2.shape[1], int)
    self.assertEqual(str(t2.shape[1]), '1')
    self.assertEqual(t2.shape[1], 1)

    # test for correctness
    t1_aten = torch.tensor([1, 0, 3, 5, 0, 6])
    t2_aten = torch.nonzero(t1_aten)
    t2_aten.fill_(1)
    self.assertEqual(t2.cpu(), t2_aten.cpu())

  def test_sizeMod(self):
    met.clear_all()

    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    # t2 has size [<=10, 2] with real size [1, 2]
    t2 = torch.nonzero(t1)
    # Create a SizeMod IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] % t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_mod"), 0)
    # Exercises SizeMod::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 1)
    self.assertEqual(str(dyn_size), '<=0')

    # t3 has size [<=10, 2] with real size [1, 2]
    t3 = torch.nonzero(t1)
    dyn_size = t2.shape[0] % t3.shape[0]
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 0)
    self.assertEqual(str(dyn_size), '<=0')

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
    # Create a SizeLt IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] < t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_lt"), 0)
    # Exercises SizeLt::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 1)

  def test_sizeGt(self):
    met.clear_all()

    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    # Create a SizeGt IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] > t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_gt"), 0)
    # Exercises SizeGt::getDynamicValue.
    dynamic_size = int(dyn_size)
    # To evaluate dynamic value (1 > 2), hence false.
    self.assertEqual(dynamic_size, 0)

  def test_sizeNe(self):
    met.clear_all()

    size1 = 5
    size2 = 2
    t1 = torch.zeros([size1, size2], device=dev)
    t1[3][0] = 1
    # t2 has size [<=10, 2]
    t2 = torch.nonzero(t1)
    # Create a SizeAdd IR node.
    # t2.shape[1] generates a SizeConstant node.
    dyn_size = t2.shape[0] != t2.shape[1]
    self.assertGreater(met.counter_value("xla::size_ne"), 0)
    # Exercises SizeNe::getDynamicValue.
    dynamic_size = int(dyn_size)
    self.assertEqual(dynamic_size, 1)

  def test_SizeEq_should_not_compile_for_identical_symints(self):
    met.clear_all()
    t1 = torch.tensor([1, 0, 3, 5, 0, 6, 7], device=dev)
    t2 = torch.nonzero(t1)
    dyn_size = t2.shape[0]
    self.assertEqual(dyn_size, dyn_size)
    # Without the code change, met.metric_data('CompileTime')[0] returns 1.
    # self.assertIsNone(met.metric_data('CompileTime'))
    # TODO(ds): Uncomment the line above after we implement 0/1 specialization.
    # The extra compilation comes from the call `set_sizes_and_strides` in XLATensorImpl::XLATensorImpl when we compare a SymInt with 0.
    self.assertEqual(met.metric_data('CompileTime')[0], 1)


if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main()
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
