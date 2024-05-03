import sys

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest


# TODO(alanwaketan): add test for views.
class InputOutputAliasesTest(unittest.TestCase):

  def test_non_view(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(4, 2, 2).to(xla_device)
    t2 = torch.randn(4, 2, 2).to(xla_device)
    xm.mark_step()
    met.clear_all()

    # check in place op aliasing.
    t3 = t1 + t2
    t1 *= 2.0
    t2 += 2.0
    xm.mark_step()

    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)

  def test_aliasing_with_cloned(self):
    xla_device = xm.xla_device()
    met.clear_all()
    t1 = torch.randn(4, 2, 2).to(xla_device)
    # t1_cloned share the same storage as t1
    t1_cloned = torch.clone(t1)
    t1 += 1
    xm.mark_step()
    # t1's storage will be alised with the ouput, need to make sure t1_cloned
    # got a new buffer and is still valid.
    torch.allclose(t1 - 1, t1_cloned)
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 1.0)

  def test_aliasing_across_mark_step(self):
    xla_device = xm.xla_device()
    met.clear_all()
    t1 = torch.randn(4, 5).to(xla_device)
    t1 += 1
    xm.mark_step()
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 1.0)
    t1 *= 100
    xm.mark_step()
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)

  def test_aliasing_with_multiple_inplace_update(self):
    BATCH_SIZE = 1
    SEQ_LEN = 128
    NUM_KV_HEADS = 16
    HEAD_SIZE = 256
    BLOCK_SIZE = 16
    DTYPE = torch.bfloat16
    num_blocks = 1024
    device = xm.xla_device()
    key = torch.randn(
        BATCH_SIZE * SEQ_LEN,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=DTYPE)
    k_cache = torch.randn(
        num_blocks * BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=DTYPE)
    slot_mapping = torch.randint(
        0, num_blocks, (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int64)
    # materalize k_cache to device data
    xm.mark_step()
    met.clear_all()
    for _ in range(10):
      k_cache.index_copy_(0, slot_mapping.flatten(), key)
    xm.mark_step()
    xm.wait_device_ops()
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 1.0)
    torch.allclose(k_cache[slot_mapping[0][0]].cpu(), key[0].cpu())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
