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
import numpy as np

import torch_xla.debug.metrics as met
from torch_xla.experimental.custom_kernel import flash_attention

xr.use_spmd()

n_devices = xr.global_runtime_device_count()
xs.set_global_mesh(xs.HybridMesh(
  ici_mesh_shape=(4, 1),
  dcn_mesh_shape=(1, 1),
  axis_names=("fsdp", "tensor"),
))

class FakeAttention(torch.nn.Module):
  def __init__(self, num_head=4, hidden_dim=256):
    super(FakeAttention, self).__init__()
    self.num_head = num_head
    self.hidden_dim = hidden_dim
    # self.d_k = hidden_dim // num_head
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
    # print(attn_output.shape)
    # B, SEQ_LEN, d_m = attn_output.shape
    # output = attn_output.reshape(B, SEQ_LEN, NUM_HEAD, self.d_k).permute(0, 2, 1, 3)
    # attn_output = self.fc(attn_output)
    return attn_output


class DummyModule(torch.nn.Module):
  def __init__(self, num_layer=3):
    super(DummyModule, self).__init__()
    self.num_layer = num_layer
    self.layers = nn.ModuleList([FakeAttention() for i in range(self.num_layer)])
  def forward(self, input):
    hidden_states = input
    xs.mark_sharding(hidden_states, xs.get_global_mesh(), ("fsdp", "tensor", None, None))
    # for layer in self.layers:
    #   hidden_states = layer(hidden_states)
    hidden_states = scan_layers(self.layers, input_data = hidden_states)
    return hidden_states



class AsStridedTest(parameterized.TestCase):
  
  @torch_xla.compile(full_graph=False)
  def scan_fa(self):
    with xm.xla_device():
      dm = DummyModule(3)
      hidden_states= torch.randn((2, 4, 256, 256)).requires_grad_()
    output = dm(hidden_states)
    loss = output.sum()
    loss.backward()
    print(hidden_states.grad)
    # print(output)
    


  # def try_dynamo(self):
  #   def test_memory_copy_consume(v):
  #     # met.clear_all()
  
  #     x = v[...,0]
  #     # x = v.permute(3, 0, 1, 2)[0]
  #     # x = torch.ops.aten.slice(v, -1, 0, 1)
  #     return x
  #     # xm.mark_step()
  #     # print(met.metrics_report())
 
  #   fn = torch.compile(test_memory_copy_consume, backend='openxla')
  #   device = xm.xla_device()
  #   met.clear_all()
  #   v = torch.randn((400,400,400,4), device=device)
  #   b = fn(v)
  #   # print(torch_xla._XLAC._get_xla_tensors_text([b]))
  #   print(met.metrics_report())
  #   # print(b)





if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  # test = unittest.main()
  # sys.exit(0 if test.result.wasSuccessful() else 1)
  test = AsStridedTest()
  test.scan_fa()