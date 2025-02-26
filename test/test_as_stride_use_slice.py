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

from functorch.compile import aot_function, make_boxed_func
from torch.library import custom_op


class AttentionLayers(torch.nn.Module):

  def __init__(self, num_head=4, hidden_dim=256):
    super(AttentionLayers, self).__init__()
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
    attn_output = self.fc(attn_output)
    return attn_output


class AttentionModule(torch.nn.Module):

  def __init__(self, num_layer=3, use_scan=False):
    super(AttentionModule, self).__init__()
    self.num_layer = num_layer
    self.use_scan = use_scan
    self.layers = nn.ModuleList(
        [AttentionLayers() for i in range(self.num_layer)])

  def forward(self, input):
    hidden_states = input
    xs.mark_sharding(hidden_states, xs.get_global_mesh(),
                     ("fsdp", "tensor", None, None))
    if not self.use_scan:
      for layer in self.layers:
        hidden_states = layer(hidden_states)
    else:
      hidden_states = scan_layers(self.layers, input_data=hidden_states)
    return hidden_states


class StridedAndSlice(torch.nn.Module):

  def __init__(self):
    super(StridedAndSlice, self).__init__()

  def forward(self, input, use_aten_slice=True):
    assert input.dim() > 1
    if not use_aten_slice:
      output = input[..., 0]
    else:
      output = torch.ops.aten.slice(input, -1, 0, 1).squeeze(-1)
    return output


def custom_op_strided_wrapper(input, use_aten_slice):
  return StridedAndSliceWithCustomOp.apply(input, use_aten_slice)


class StridedAndSliceWithCustomOp(torch.autograd.Function):

  def __init__(self):
    super(StridedAndSliceWithCustomOp, self).__init__()

  @staticmethod
  def forward(ctx, input, use_aten_slice=True):
    assert input.dim() > 1
    ctx.save_for_backward(input)
    ctx.use_aten_slice = use_aten_slice
    ctx.needs_input_grad = input.requires_grad
    return custom_strided_and_slice_forward(input, use_aten_slice)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):
    input, = ctx.saved_tensors
    needs_input_grad = ctx.needs_input_grad
    use_aten_slice = ctx.use_aten_slice
    assert input.dim() > 1
    grad_input = custom_strided_and_slice_backward(grad_output, input,
                                                   use_aten_slice,
                                                   needs_input_grad)
    return grad_input, None


@custom_op("xla::custom_strided_and_slice_forward", mutates_args=())
def custom_strided_and_slice_forward(input: torch.Tensor,
                                     use_aten_slice: bool) -> torch.Tensor:
  assert input.dim() > 1
  i = input.clone()
  if not use_aten_slice:
    output = i[..., 0]
  else:
    output = torch.ops.aten.slice(i, -1, 0, 1).squeeze(-1)
  return output


@custom_strided_and_slice_forward.register_fake
def custom_strided_and_slice_forward_fake(input: torch.Tensor,
                                          use_aten_slice: bool) -> torch.Tensor:
  return torch.empty_like(input[..., 0])


@custom_op("xla::custom_strided_and_slice_backward", mutates_args=())
def custom_strided_and_slice_backward(grad_output: torch.Tensor,
                                      input: torch.Tensor, use_aten_slice: bool,
                                      needs_input_grad: bool) -> torch.Tensor:
  assert input.dim() > 1
  raise NotImplementedError("This should not be called")


@custom_strided_and_slice_backward.register_fake
def custom_strided_and_slice_backward_fake(
    grad_output: torch.Tensor, input: torch.Tensor, use_aten_slice: bool,
    needs_input_grad: bool) -> torch.Tensor:
  return torch.empty_like(input)


#############Test Class Begins#################


class AsStridedTest(parameterized.TestCase):

  def pure_strided_wrapper(self, use_xla, use_aten_slice):
    ss = StridedAndSlice().to("cpu")
    input = torch.randn((2, 4, 256, 256), device="cpu").requires_grad_()
    if use_xla:
      ss.to(xm.xla_device())
      input = input.to(xm.xla_device())
    return ss(input, use_aten_slice)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU")
  @parameterized.named_parameters(
      ("use_aten_slice_True", True),
      ("use_aten_slice_False", False),
  )
  def test_pure_as_strided(self, use_aten_slice):
    """compare torch native against xla aten.slice/aten.as_strided"""
    torch.manual_seed(12)
    cpu_output = self.pure_strided_wrapper(
        use_xla=False, use_aten_slice=use_aten_slice)
    torch.manual_seed(12)
    xla_output = self.pure_strided_wrapper(
        use_xla=True, use_aten_slice=use_aten_slice)
    torch.testing.assert_close(cpu_output, xla_output.cpu())

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU")
  @parameterized.named_parameters(
      ("use_aten_slice_True", True),
      ("use_aten_slice_False", False),
  )
  def test_custom_ops_as_strided(self, use_aten_slice):

    def compiler(gm, _):
      return make_boxed_func(gm)

    compiler_func = aot_function(
        custom_op_strided_wrapper, fw_compiler=compiler)
    torch.manual_seed(12)
    torch_xla.manual_seed(12)
    input_cpu = torch.randn((2, 2, 3, 3), requires_grad=True)
    input_xla = input_cpu.clone().detach().requires_grad_()

    cpu_output = compiler_func(input_cpu, use_aten_slice=use_aten_slice)
    torch_xla.sync()

    input_xla = input_xla.to(xm.xla_device())
    xla_output = compiler_func(input_xla, use_aten_slice=use_aten_slice)
    torch_xla.sync()
    torch.testing.assert_close(cpu_output, xla_output.cpu())


class ScanFlashAttentionTest(parameterized.TestCase):

  def fake_fa_wrapper(self, use_scan):
    with xm.xla_device():
      dm = AttentionModule(3, use_scan=use_scan)
      hidden_states = torch.randn((2, 4, 256, 256)).requires_grad_()
    hidden_states.retain_grad()
    output = dm(hidden_states)
    return output

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU")
  @parameterized.named_parameters(("use_scan_True", True),
                                  ("use_scan_False", False))
  def test_scan_layer_aot(self, use_scan):
    output = self.fake_fa_wrapper(use_scan)
    torch_xla.sync()
    # TODO(https://github.com/pytorch/xla/issues/8742): Fix NaN
    # self.assertFalse(torch.isnan(output).any())


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)

  xr.use_spmd()
  n_devices = xr.global_runtime_device_count()
  xs.set_global_mesh(
      xs.HybridMesh(
          ici_mesh_shape=(n_devices, 1),
          dcn_mesh_shape=(1, 1),
          axis_names=("fsdp", "tensor"),
      ))

  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
