import sys
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla._dynamo.dynamo_bridge import alias_with_buffer_donor_config


class TestBufferDonationUtil(unittest.TestCase):

  def test_hash_with_buffer_donor(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    res = torch.cos(input)
    hash_no_donor = torch_xla._XLAC._get_graph_hash([res])
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # without the alias_with_buffer_donor_config context, buffer donor will be ignored,
    # so we still expect the hash to be the same.
    hash_with_donor = torch_xla._XLAC._get_graph_hash([res])
    self.assertEqual(hash_no_donor, hash_with_donor)

    with alias_with_buffer_donor_config() as saved_config:
      hash_with_donor_and_context = torch_xla._XLAC._get_graph_hash([res])
    self.assertNotEqual(hash_no_donor, hash_with_donor_and_context)


class TestDynamoBufferDonationAliasingWithCustomOp(unittest.TestCase):

  def dummy_inplace_mul(self, input):
    # always donate input buffer
    torch.ops.xla.dynamo_set_buffer_donor_(input, True)
    input *= 1.1
    return

  def dummy_mul(self, input):
    # always donate input buffer
    torch.ops.xla.dynamo_set_buffer_donor_(input, True)
    return input * 1.1

  def test_manual_buffer_donation(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = input.cpu().to(device)
    dummy_inplace_mul_compiled = torch.compile(
        self.dummy_inplace_mul, backend='openxla')

    met.clear_all()
    dummy_inplace_mul_compiled(input)
    self.assertIn('XlaSetBufferDonation', met.counter_names())
    # Dynamo will call `dynamo_set_buffer_donor_` once on the faketensor and call
    # it again on real tensor in our dynamo bridge.
    self.assertEqual(met.counter_value('XlaSetBufferDonation'), 2)
    torch.allclose(input_cloned.cpu() * 1.1, input.cpu())

  def test_manual_buffer_donation_for_non_inplce_op(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = input.cpu().to(device)
    dummy_mul_compiled = torch.compile(self.dummy_mul, backend='openxla')

    met.clear_all()
    res = dummy_mul_compiled(input)
    # check input's buffer has been aliased.
    xm.wait_device_ops()
    # make sure buffer donation setting is correctly updated
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))
    self.assertIn('XlaSetBufferDonation', met.counter_names())
    # Dynamo will call `dynamo_set_buffer_donor_` once on the faketensor and call
    # it again on real tensor in our dynamo bridge.
    self.assertEqual(met.counter_value('XlaSetBufferDonation'), 2)
    self.assertIn('Data Handle: Deleted',
                  torch_xla._XLAC._get_xla_tensor_debug_info(input))
    torch.allclose(input_cloned.cpu() + 1, res.cpu())

  def test_manual_buffer_donation_for_inplce_op_repeat(self):
    # use a different function than above dummy add otherwise XLA won't recompile
    def dummy_inplace(input):
      # always donate input buffer
      torch.ops.xla.dynamo_set_buffer_donor_(input, True)
      input += (0.5 * torch.sin(input))

    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = input.cpu().to(device)
    dummy_inplace_add_compiled = torch.compile(dummy_inplace, backend='openxla')
    torch_xla.sync()
    met.clear_all()

    for _ in range(100):
      dummy_inplace_add_compiled(input)
    # should_donate_buffer field is attached to the buffer and won't be inherited to
    # the output buffer(unless execution is a no-op). However dynamo don't track this
    # field so it will keep executing the graph with input buffer being aliased.
    self.assertFalse(torch_xla._XLAC._get_buffer_donation(input))
    # there shouldn't be any recompilation even `should_donate_buffer` field changed after
    # first execution. This is because Dynamo does not trace this internal field for xla.
    self.assertEqual(met.metric_data('CompileTime')[0], 1)


class TestDynamoBufferDonationAliasing(unittest.TestCase):

  def dummy_inplace_add(self, input):
    input += 1
    return

  def dummy_add(self, input):
    return input + 1

  def test_manual_buffer_donation(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = input.cpu().to(device)
    dummy_inplace_add_compiled = torch.compile(
        self.dummy_inplace_add, backend='openxla')

    met.clear_all()
    # input is a device_data, we should be able to set the buffer donation field.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # make sure buffer donation setting is correctly updated
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))
    self.assertIn('XlaSetBufferDonation', met.counter_names())
    self.assertEqual(met.counter_value('XlaSetBufferDonation'), 1)
    dummy_inplace_add_compiled(input)
    torch.allclose(input_cloned.cpu() + 1, input.cpu())
    self.assertFalse(torch_xla._XLAC._get_buffer_donation(input))

  def test_manual_buffer_donation_for_non_inplce_op(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = input.cpu().to(device)
    dummy_add_compiled = torch.compile(self.dummy_add, backend='openxla')

    met.clear_all()
    # input is a device_data, we should be able to set the buffer donation field.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # make sure buffer donation setting is correctly updated
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))
    self.assertIn('XlaSetBufferDonation', met.counter_names())
    self.assertEqual(met.counter_value('XlaSetBufferDonation'), 1)

    res = dummy_add_compiled(input)
    # check input's buffer has been aliased.
    xm.wait_device_ops()
    self.assertIn('Data Handle: Deleted',
                  torch_xla._XLAC._get_xla_tensor_debug_info(input))
    torch.allclose(input_cloned.cpu() + 1, res.cpu())

  def test_manual_buffer_donation_for_inplce_op_repeat(self):
    # use a different function than above dummy add otherwise XLA won't recompile
    def dummy_inplace(input):
      input += (0.3 * torch.cos(input))

    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    input_cloned = input.cpu().to(device)
    dummy_inplace_add_compiled = torch.compile(dummy_inplace, backend='openxla')
    torch_xla.sync()
    met.clear_all()
    # input is a device_data, we should be able to set the buffer donation field.
    self.assertTrue(torch_xla._XLAC._set_buffer_donation(input, True))
    # make sure buffer donation setting is correctly updated
    self.assertTrue(torch_xla._XLAC._get_buffer_donation(input))

    for _ in range(100):
      dummy_inplace_add_compiled(input)
    # should_donate_buffer field is attached to the buffer and won't be inherited to
    # the output buffer(unless execution is a no-op). However dynamo don't track this
    # field so it will keep executing the graph with input buffer being aliased.
    self.assertFalse(torch_xla._XLAC._get_buffer_donation(input))
    # there shouldn't be any recompilation even `should_donate_buffer` field changed after
    # first execution. This is because Dynamo does not trace this internal field for xla.
    self.assertEqual(met.metric_data('CompileTime')[0], 1)

  def test_buffer_donation_on_non_data_tensor(self):
    device = xm.xla_device()
    input = torch.randn(5, 5).to(device)
    res = input + 1

    met.clear_all()
    # res now points to a `Add` IR, only data's buffer can be aliased
    self.assertFalse(torch_xla._XLAC._set_buffer_donation(res, True))
    self.assertFalse(torch_xla._XLAC._get_buffer_donation(res))
    self.assertNotIn('XlaSetBufferDonation', met.counter_names())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
