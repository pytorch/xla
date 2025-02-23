import logging
import sys
import unittest
from absl.testing import parameterized

import torch
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
from torch_xla._internal import tpu
from torch_xla.experimental.scan_layers import scan_layers
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.custom_kernel import flash_attention


class FakeAttention(torch.nn.Module):
  def __init__(self, num_head=4, hidden_dim=256):
    super(FakeAttention, self).__init__()
    self.num_head = num_head
    self.hidden_dim = hidden_dim
    self.fc = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, input):
    # query_states: [B, NUM_HEAD, SEQ_LEN, d_k]
    # attn_output: [B, SEQ_LEN, d_m], dm = dk * NUM_HEAD
    query_states = input.clone()
    key_states = input.clone()
    value_states = input.clone()
    attn_output = flash_attention(
        query_states,
        key_states,
        value_states,
        causal=True,
        partition_spec=("fsdp", "tensor", None, None),
      )
    # below statement is unnecessary for testing the scan and flash attention
    # kernel
    attn_output = self.fc(attn_output)
    return attn_output


class DummyModule(torch.nn.Module):
  def __init__(self, num_layer=3, use_scan=False):
    super(DummyModule, self).__init__()
    self.num_layer = num_layer
    self.use_scan = use_scan
    self.layers = nn.ModuleList([FakeAttention() for i in range(self.num_layer)])
  def forward(self, input):
    hidden_states = input
    xs.mark_sharding(hidden_states, xs.get_global_mesh(), ("fsdp", "tensor", None, None))
    if not self.use_scan:
      for layer in self.layers:
        hidden_states = layer(hidden_states)
    else:
      hidden_states = scan_layers(self.layers, input_data = hidden_states)
    return hidden_states


class AsStridedTest(parameterized.TestCase):
  
  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @parameterized.parameters(
      True,
      False,
  )
  @torch_xla.compile(full_graph=False)
  def test_scan_layer_aot(self, use_scan):
    with xm.xla_device():
      dm = DummyModule(3, use_scan=use_scan)
      hidden_states= torch.randn((2, 4, 256, 256)).requires_grad_()
    hidden_states.retain_grad()
    output = dm(hidden_states)
    loss = output.sum()
    loss.backward()
    xm.mark_step()
    print(hidden_states.grad)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
    
  xr.use_spmd()
  n_devices = xr.global_runtime_device_count()
  xs.set_global_mesh(xs.HybridMesh(
    ici_mesh_shape=(n_devices, 1),
    dcn_mesh_shape=(1, 1),
    axis_names=("fsdp", "tensor"),
  ))

  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
