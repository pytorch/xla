import logging
import sys
import unittest
from absl.testing import parameterized

import torch
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
from torch_xla.experimental.scan_layers import scan_layers
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.custom_kernel import flash_attention


class AttentionModule(torch.nn.Module):

  def __init__(self, has_model_weight=True, num_head=4, hidden_dim=256):
    super(AttentionModule, self).__init__()
    self.has_model_weight = has_model_weight
    if has_model_weight:
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
        partition_spec=("fsdp", None, None, None),
    )
    if self.has_model_weight:
      attn_output = self.fc(attn_output)
    return attn_output


class AttentionLayers(torch.nn.Module):

  def __init__(self, has_model_weight=True, num_layer=3, use_scan=False):
    super(AttentionLayers, self).__init__()
    self.num_layer = num_layer
    self.use_scan = use_scan
    self.has_model_weight = has_model_weight
    self.layers = nn.ModuleList([
        AttentionModule(has_model_weight=has_model_weight)
        for i in range(self.num_layer)
    ])

  def forward(self, input):
    hidden_states = input
    xs.mark_sharding(hidden_states, xs.get_global_mesh(),
                     ("fsdp", None, None, None))
    if not self.use_scan:
      for layer in self.layers:
        hidden_states = layer(hidden_states)
    else:
      hidden_states = scan_layers(self.layers, input_data=hidden_states)
    return hidden_states


class ScanFlashAttentionTest(parameterized.TestCase):

  def fake_fa_wrapper(self, has_model_weight, use_scan):
    torch.manual_seed(12)
    torch_xla.manual_seed(12)
    hidden_states = torch.randn((2, 4, 256, 256)).requires_grad_().to('xla')
    with xm.xla_device():
      attention_layers = AttentionLayers(
          has_model_weight, num_layer=3, use_scan=use_scan)
    hidden_states.retain_grad()
    output = attention_layers(hidden_states)
    return output

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU")
  def test_scan_flash_attention_against_for_loop(self):
    for_loop_output = self.fake_fa_wrapper(
        has_model_weight=True, use_scan=False)
    torch_xla.sync()
    scan_output = self.fake_fa_wrapper(has_model_weight=True, use_scan=True)
    torch_xla.sync()
    torch.testing.assert_close(
        for_loop_output.cpu(), scan_output.cpu(), atol=1e-3, rtol=1e-3)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU")
  @parameterized.named_parameters(("has_model_weight_True", True),
                                  ("has_model_weight_False", False))
  def test_scan_weight_layer_aot(self, has_model_weight_scan):
    output = self.fake_fa_wrapper(
        has_model_weight=has_model_weight_scan, use_scan=False)
    torch_xla.sync()
    # TODO(https://github.com/pytorch/xla/issues/8753): Fix assertion
    # torch.manual_seed(12)
    # torch_xla.manual_seed(12)
    # scan_output = self.fake_fa_wrapper(
    #     has_model_weight=has_model_weight_scan, use_scan=True)
    # torch_xla.sync()
    # torch.testing.assert_close(output.cpu(), scan_output.cpu())


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)

  xr.use_spmd()
  n_devices = xr.global_runtime_device_count()
  xs.set_global_mesh(xs.get_1d_mesh("fsdp"))

  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
