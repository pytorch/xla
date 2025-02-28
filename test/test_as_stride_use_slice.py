import logging
import sys
import unittest
from absl.testing import parameterized

import torch
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
import torch_xla.distributed.spmd as xs

from functorch.compile import aot_function, make_boxed_func
from torch.library import custom_op


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


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
