import os
import sys

import torch
import torch_xla
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import unittest


class DummyModel(nn.Module):

  def __init__(self):
    super(DummyModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.fc1 = nn.Linear(250, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = x.view(-1, 250)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


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

  def test_check_tensor_need_materialization(self):
    xla_device = xm.xla_device()
    t1 = torch.randn(20, 5)
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    t1 = t1.to(xla_device)
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    t2 = t1 * 2
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [False])
    assert (torch_xla._XLAC._check_tensor_need_materialization([t2]) == [True])
    t1.mul_(100)
    assert (torch_xla._XLAC._check_tensor_need_materialization([t1]) == [True])

  def test_get_graph_hash(self):
    xla_device = xm.xla_device()
    xla_input = torch.randn(20, 1, 14, 14).to(xla_device)
    xla_dummy_model = DummyModel().to(xla_device)
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

  def test_run_cached_graph(self):
    xla_device = xm.xla_device()
    xla_input = torch.randn(64, 256, 14, 14).to(xla_device)
    xla_dummy_model = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.ReLU(),
        ).to(xla_device)
    # xla_out = xla_dummy_model(xla_input)
    xla_out = xla_dummy_model(xla_input)
    import pdb
    pdb.set_trace()
    hash = torch_xla._XLAC._get_graph_hash([xla_out])
    # Force trigger an execution to cache this computation.
    torch_xla._XLAC._xla_sync_multi([xla_out], [])
    # TODO(JackCaoG): need to include other parameters
    torch_xla._XLAC._run_cached_graph(hash, [xla_input])


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
