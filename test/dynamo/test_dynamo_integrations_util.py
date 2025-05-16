import os
import sys

import torch
import torch_xla
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest

dummy_model = torch.nn.Sequential(
    torch.nn.Conv2d(256, 256, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(256, 256, 1),
    torch.nn.ReLU(),
)


class PybindTest(unittest.TestCase):

  def test_get_tensors_xla_device_data_node(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(20, 5).to(xla_device)
    t2 = torch.randn(20, 5).to(xla_device)
    t3 = t2 + t1
    t4 = t3 * t2
    res_pair = torch_xla._XLAC._get_tensors_xla_device_data_node([t4])
    # res_pair[0] is tensor_ids and res_pair[1] are actual at::tensors that wrap around the
    # XLATensor and XLAData
    assert (len(res_pair[0]) == len(res_pair[1]))
    # only t1 and t2 are device data
    assert (len(res_pair[0]) == 2)
    expected_data_handles = sorted(
        torch_xla._XLAC._get_tensors_handle([t1, t2]))
    real_data_handles = sorted(torch_xla._XLAC._get_tensors_handle(res_pair[1]))
    assert (expected_data_handles == real_data_handles)
    expected_tensor_ids = sorted([
        torch_xla._XLAC._xla_get_tensor_id(t1),
        torch_xla._XLAC._xla_get_tensor_id(t2)
    ])
    assert (expected_tensor_ids == sorted(res_pair[0]))

  def test_get_base_seed_as_tensor(self):
    device = xm.xla_device()
    xm.set_rng_state(23, str(device))
    base_seed = torch_xla._XLAC._get_base_seed_as_tensor(str(device)).item()
    self.assertEqual(23, base_seed)

  def test_get_seed_info_id(self):
    self.assertEqual(torch_xla._XLAC._get_seed_info_id(), -127389)

  def test_check_tensor_need_materialization(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(20, 5)
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    t1 = t1.to(xla_device)
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    # call `torch_xla.sync()` to clear pending irs on t1. This should test the
    # case where XLATensor has a `XLAData` instead of a `DeviceData` IR.
    torch_xla.sync()
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    t2 = t1 * 2
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    assert (torch_xla._XLAC._check_tensor_need_materialization([t2]) == [True])
    t1.mul_(100)
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [True])

  def test_get_graph_hash(self):
    xla_device = xm.xla_device()
    xla_input = torch.randn(64, 256, 14, 14).to(xla_device)
    xla_dummy_model = dummy_model.to(xla_device)
    xla_out = xla_dummy_model(xla_input)
    hash = torch_xla._XLAC._get_graph_hash([xla_out])
    # Calling the _get_graph_hash twice should omit the same hash
    assert (hash == torch_xla._XLAC._get_graph_hash([xla_out]))

    # Inplace update the xla_out and now hash should be updated
    xla_out += 1
    assert (hash != torch_xla._XLAC._get_graph_hash([xla_out]))

    # Repeat the same computation on input with same shape, hash should be the same
    xla_input2 = torch.randn(20, 1, 14, 14).to(xla_device)
    xla_out_2 = xla_dummy_model(xla_input)
    assert (hash == torch_xla._XLAC._get_graph_hash([xla_out_2]))

  def test_clear_pending_irs(self):
    xla_device = xm.xla_device()
    torch_xla.sync()
    t1 = torch.randn(20, 5).to(xla_device)
    t2 = torch.randn(20, 5).to(xla_device)
    t3 = t2 + t1
    t4 = t3 * t2
    met.clear_metrics()
    torch_xla._XLAC._xla_sync_multi([t4], devices=[], wait=True)
    # only t4 is materialized
    self.assertIn("aten::add", torch_xla._XLAC._get_xla_tensors_text([t3]))
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)
    torch_xla._XLAC._clear_pending_irs(str(xla_device))
    self.assertNotIn("aten::add", torch_xla._XLAC._get_xla_tensors_text([t3]))
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)
    torch_xla.sync()
    # `torch_xla.sync()` should not incur new execution
    self.assertEqual(met.metric_data('ExecuteTime')[0], 1)

  def test_run_cached_graph(self):
    xla_device = xm.xla_device()
    xla_input = torch.randn(64, 256, 14, 14).to(xla_device)
    xla_dummy_model = dummy_model.to(xla_device)
    xla_out = xla_dummy_model(xla_input)
    hash = torch_xla._XLAC._get_graph_hash([xla_out])
    # Warm up the cache.
    torch_xla._XLAC._xla_warm_up_cache([xla_out], [])

    # It is the caller of `run_cached_graph`'s job to make sure the input order
    # matches the graph input order. Upstream dynamo has a more completed
    # logic of tracking input orders using tensor id. In this test I am going
    # to hack it since above model is simple.
    expected_input = [t for t in xla_dummy_model.parameters()]
    expected_input.reverse()
    expected_input.append(xla_input)
    hash_out = torch_xla._XLAC._run_cached_graph(hash, expected_input)
    assert (len(hash_out) == 1)
    assert (hash_out[0].equal(xla_out))

    expected_input_cpu = [t.cpu() for t in expected_input]
    hash_out = torch_xla._XLAC._run_cached_graph(hash, expected_input_cpu)
    assert (len(hash_out) == 1)
    assert (hash_out[0].equal(xla_out))

    assert ('RunCachedGraphInputData' in met.metric_names())
    assert ('RunCachedGraphOutputData' in met.metric_names())


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
