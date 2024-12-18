import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest
import copy


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

  def test_aliasing_across_custom_inplace(self):
    xla_device = xm.xla_device()
    met.clear_all()
    t1 = torch.randn(4, 5).to(xla_device)
    t1 *= t1
    xm.mark_step()
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 1.0)
    xm.optimization_barrier_([t1])
    t1 *= 100
    xm.mark_step()
    self.assertEqual(met.metric_data("InputOutputAliasCount")[1], 2.0)

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

  def test_grad_accum(self):

    class MLP(nn.Module):

      def __init__(self, input_size=28 * 28, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)

      def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def try_grad_accum(model, device, train_x, train_label, accum_steps):
      loss_fn = nn.NLLLoss()
      train_x = train_x.to(device)
      train_label = train_label.to(device)
      model.zero_grad()
      for i in range(accum_steps):
        output = model(train_x)
        t_loss = loss_fn(output, train_label)
        t_loss.backward()
        xm.mark_step()
      return [p.grad.to('cpu').numpy() for p in model.parameters()]

    dev = xm.xla_device()
    train_x_sample = torch.rand((1, 28 * 28))
    train_label_sample = torch.tensor([5])
    c_model = MLP().to('cpu')
    t_model = copy.deepcopy(c_model).to(dev)
    t_model.train()
    c_model.train()
    accum_steps = 4
    c_grads_5 = try_grad_accum(c_model, 'cpu', train_x_sample,
                               train_label_sample, accum_steps)
    met.clear_metrics()
    t_grads_5 = try_grad_accum(t_model, dev, train_x_sample, train_label_sample,
                               accum_steps)
    torch.testing.assert_close(t_grads_5, c_grads_5, rtol=3e-2, atol=1e-3)
    graph_count, alias_count, _ = met.metric_data("InputOutputAliasCount")
    assert (
        graph_count == 2
    ), f"Expect 2 graphs for gradient accumulation test, got {graph_count}"
    assert (
        alias_count == 1.0
    ), f"Expect 1 input-output alias pair for gradient accumulation, got {alias_count}"

  def test_separate_graphs(self):
    """
    Test that paramater aliasing differences should produce different graphs.
    """
    xla_device = xm.xla_device()
    t0 = torch.tensor([1], device=xla_device)
    t1 = torch.tensor([2], device=xla_device)
    xm.mark_step()

    t1.add_(t0)
    xm.mark_step()

    # This needs to be a separate graph, otherwise t1 can be corrupted
    # or result in PJRT error.
    t2 = t1 + t0
    xm.mark_step()

    self.assertEqual(t1.item(), 3)

  def test_xm_save_no_aliasing(self):
    """
    Test that xm.save() does not perform aliasing.
    """
    xla_device = xm.xla_device()
    t0 = torch.tensor([1], device=xla_device)
    t1 = torch.tensor([2], device=xla_device)
    xm.mark_step()

    t2 = t0 + t1
    t1.add_(1)

    # Save the new value of t1 should not result in the old value
    # being donated...
    xm.save(t1, os.devnull)

    # otherwise this mark_step could crash, or compute the wrong value
    # for t2.
    xm.mark_step()

    self.assertEqual(t2.item(), 3)

  def test_device_data_cache_no_aliasing(self):
    """
    Test that device data in DataCache are not aliased.
    """
    xla_device = xm.xla_device()

    t0 = torch.tensor(42, device=xla_device)
    # drops the read-only bit on t0's device_data
    xm.mark_step()

    # cached value of 42 is donated
    t0.add_(1)
    xm.mark_step()

    # t1 get the cached device_data, which was donated
    t1 = torch.tensor(42, device=xla_device)
    xm.mark_step()

    t1.add_(1)
    # XLA crashes here because parameter is donated buffer...
    xm.mark_step()

    # ...if it doesn't crash, the value here would be 44.
    self.assertEqual(t1.item(), 43)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
