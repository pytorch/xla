import copy
import sys
import unittest

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
local_bs = 4096 * 8
input_dim = 4096
hidden_dim = 8192


class M(torch.nn.Module):

  def __init__(self):
    super(M, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.fc1(x)


class M2(torch.nn.Module):

  def __init__(self, mesh):
    super(M2, self).__init__()
    self.fc1 = torch.nn.Linear(hidden_dim, input_dim, bias=False)
    self.mesh = mesh

  def forward(self, x):
    return self.fc1(x)


class DisableSPMDTest(unittest.TestCase):

  def _init_ddp_model(self, ddp_model):
    xr.disable_spmd()
    model_replicas = []
    num_local_devices = xr.addressable_runtime_device_count()
    for i in range(num_local_devices):
      curr_device = f'xla:{i}'
      model_replicas.append(copy.deepcopy(ddp_model).to(curr_device))
    return model_replicas

  def _init_spmd_model(self, spmd_model):
    xr.use_spmd()
    device = xm.xla_device()
    spmd_model = spmd_model.to(device)
    return spmd_model

  def _run_single_process_ddp(self, ddp_model_replicas, input_shape):
    xr.disable_spmd()
    num_local_devices = xr.addressable_runtime_device_count()
    outputs = []
    # Move model and input data to each local device.
    for i in range(num_local_devices):
      curr_device = f'xla:{i}'
      input = torch.randn(input_shape).to(curr_device)
      with torch.no_grad():
        output = ddp_model_replicas[i](input)
        input = input.cpu()
        outputs.append(output)
    xm.wait_device_ops()
    torch_xla.sync(True)
    return outputs

  def _run_spmd(self, spmd_model, input_shape, mesh):
    xr.use_spmd(force_tensors_on_spmd_device=False)
    device = xm.xla_device()
    spmd_input = torch.randn(input_shape).to(device)
    spmd_input = xs.mark_sharding(spmd_input, mesh, (0, None))
    with torch.no_grad():
      spmd_out = spmd_model(spmd_input)
    xm.wait_device_ops()
    torch_xla.sync(True)
    return spmd_out

  def test_disable_spmd(self):

    num_iter = 3
    single_ddp_outputs = []
    spmd_outputs = []
    ddp_input_shape = (local_bs, input_dim)

    num_global_devices = xr.global_runtime_device_count()
    spmd_input_shape = (local_bs * num_global_devices, hidden_dim)
    mesh_shape = (num_global_devices, 1)
    device_ids = np.array(range(num_global_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))

    model = M()
    ddp_model_replicas = self._init_ddp_model(model)

    spmd_model = M2(mesh)
    spmd_model = self._init_spmd_model(spmd_model)

    # 8: 4 transfers for model, 4 transfers for ddp input
    # 14: Increse by 6:
    #     2 transfers in SPMD run (1 for model in the 1st run, 1 for input)
    #     4 transfers for ddp input
    # 19: Increse by 5:
    #     1 transfers for SPMD input
    #     4 transfers for ddp input
    expected_transfer_to_device_counter = [8, 14, 19]

    for i in range(num_iter):
      single_ddp_output = self._run_single_process_ddp(
          ddp_model_replicas, input_shape=ddp_input_shape)
      single_ddp_outputs.append(single_ddp_output)
      self.assertEqual(
          met.metric_data('TransferToDeviceTime')[0],
          expected_transfer_to_device_counter[i])
      spmd_output = self._run_spmd(spmd_model, spmd_input_shape, mesh)
      spmd_outputs.append(spmd_output)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
