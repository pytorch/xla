import glob
import os
from absl.testing import absltest

import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr


class TestDataTransfer(absltest.TestCase):

  def setUp(self):
    met.clear_all()

  def test_h2d_tensor_no_copy(self):
    t = torch.zeros(10, 10)
    t = t.to('xla')
    self.assertNotIn('AtenSourceTensorCopy', met.counter_names())

  def test_h2d_tensor_copy(self):
    # Non-contiguous tensor will trigger a copy.
    t = torch.zeros(10, 10).transpose(0, 1)
    t = t.to('xla')
    self.assertIn('AtenSourceTensorCopy', met.counter_names())
    self.assertEqual(met.counter_value('AtenSourceTensorCopy'), 1)

  @absltest.skipUnless(xr.device_type() == 'CUDA',
                       "This test only runs on CUDA.")
  def test_h2d_tensor_cuda(self):
    # If a torch tensor is on cuda, now it will be copied to CPU
    # before sending to GPU via PJRT.
    t = torch.zeros(10, 10).to('cuda')
    t = t.to('xla')
    self.assertIn('AtenSourceTensorCopy', met.counter_names())
    self.assertEqual(met.counter_value('AtenSourceTensorCopy'), 1)


if __name__ == '__main__':
  absltest.main()
