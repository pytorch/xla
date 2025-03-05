from absl.testing import absltest
import torch
import torch_xla
from torch_xla.core.xla_builder import create_placeholder_tensor
import torch_xla.debug.metrics as met
import re


class TestPlaceholder(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch_xla._XLAC._xla_set_enable_alias_with_buffer_donor_config(True)

  def test_create_placeholder(self):
    for shape, dtype in zip(
        ((1, 2), (2, 3, 4), (3, 4, 5, 6)),
        (torch.float32, torch.bfloat16, torch.int8),
    ):
      p = create_placeholder_tensor(shape, dtype)
      assert isinstance(p, torch.Tensor)
      assert p.device == torch_xla.device()
      self.assertEqual(p.dtype, dtype)
      self.assertEqual(p.shape, shape)
      self.assertTrue(torch_xla._XLAC._is_placecholder(p))

  def test_read_value_crashes(self):
    p = create_placeholder_tensor((1,), torch.bfloat16)
    with self.assertRaises(RuntimeError):
      p.cpu()

  def test_trace_graph(self):
    met.clear_all()
    self.assertFalse(met.metric_data("TransferToDeviceTime"))

    p1 = create_placeholder_tensor((2, 3), torch.bfloat16)
    a = torch.sin(p1)
    p2 = create_placeholder_tensor((3, 4), torch.bfloat16)
    # We use p1 once and p2 twice. But the graph should still only have two parameters.
    b = (a @ p2) @ p2.T
    ir: str = torch_xla._XLAC._get_xla_tensors_text([b])
    self.assertEqual(ir.count("xla::device_data()"), 2)
    self.assertEqual(ir.count("bf16[3,4]{1,0} xla::device_data()"), 1)
    self.assertEqual(ir.count("bf16[2,3]{1,0} xla::device_data()"), 1)
    hlo: str = torch_xla._XLAC._get_xla_tensors_hlo([b])
    regex = r'\(p.*: bf16\[3,4\], p.*: bf16\[2,3\]\) -> \(bf16\[2,3\]\)'
    assert re.search(regex, hlo) is not None

    # There should be no buffers transferred to the device during tracing
    self.assertFalse(met.metric_data("TransferToDeviceTime"))

  def test_placeholder_handle_unique(self):
    p1 = create_placeholder_tensor((1,), torch.bfloat16)
    p2 = create_placeholder_tensor((1,), torch.bfloat16)
    h1, h2 = torch_xla._XLAC._get_tensors_handle([p1, p2])
    self.assertNotEqual(h1, h2)

  def test_cannot_get_handle_from_deleted_pjrt_buffer(self):
    xla_device = torch_xla.device()
    t0 = torch.randn(4, 2, 2).to(xla_device)
    t1 = torch.randn(4, 2, 2).to(xla_device)
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(t0, True))
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(t0))
    _ = t0 + t1
    torch_xla.sync(wait=True)

    self.assertTrue(torch_xla._XLAC._is_placecholder(t0))
    with self.assertRaises(RuntimeError, msg='is deleted'):
      torch_xla._XLAC._get_tensors_handle([t0])


if __name__ == "__main__":
  absltest.main()
