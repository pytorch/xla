import torch
import torch.nn.functional as F
import torch_xla
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "quantized_matmul(Tensor x, Tensor w, Tensor scale, int? blocksize=-1, bool? quantize_activation=False) -> Tensor"
)


def _check_per_channel_quant_weight_dtype_shapes(input_dim, output_dim, w,
                                                 w_scaler):
  assert w.dtype == torch.int8, f"Weight dtype is expected to be torch.int8, got {w.dtype}."
  assert w.dim(
  ) == 2, f"Weight tensor is expected to be 2D, got {w.dim()}D Tensor."
  assert output_dim == w.shape[0] and input_dim == w.shape[
      1], f"Weight shape is expected to be [output_dim, input_dim], output_dim: {output_dim}, input_dim: {input_dim}, but got {w.shape}."
  assert w_scaler.dim() == 1 and w_scaler.shape[0] == w.shape[
      0], f"weight scaler shape is expect to be [out_channel,], got {w_scaler.shape}, weight shape {w.shape}."


@impl(XLA_LIB, "quantized_matmul", "XLA")
def quantized_matmul_xla(x: torch.Tensor,
                         w: torch.Tensor,
                         scaler: torch.Tensor,
                         blocksize: int = -1):
  """Quantized Matrix Multiply op on XLA devices.

  Args:
      x: torch.Tensor - Activation of Matmul [..., in_channel].
      w: torch.Tensor - Weight Tensor.
         per-channel quant: torch.int8 x [out_channel, in_channel].
      scaler: torch.Tensor - Weight scaler.
         per-channel quant: [out_channel,].
      blocksize: blocksize for blockwise quantization, -1 for per-channel quantization.
  """
  assert blocksize == -1, "blockwise quantization is not supported yet."
  # Per-channel quant.
  _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0], w,
                                               scaler)
  return F.linear(x, w) * scaler


@impl(XLA_LIB, "quantized_matmul", "CompositeExplicitAutograd")
def quantized_matmul(x: torch.Tensor,
                     w: torch.Tensor,
                     scaler: torch.Tensor,
                     blocksize: int = -1):
  assert blocksize == -1, "blockwise quantization is not supported yet."
  # Per-channel quant.
  _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0], w,
                                               scaler)
  w = w.to(x.dtype)
  return torch.mul(F.linear(x, w), scaler)


class XlaQuantizedLinear(torch.nn.Module):

  def __init__(self, input_dim, output_dim, blocksize=-1):
    super().__init__()
    assert blocksize == -1, "Only per-channel quantization is supported."
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.blocksize = blocksize
    self.register_buffer('weight',
                         torch.zeros(output_dim, input_dim).to(torch.int8))
    self.register_buffer('weight_scaler', torch.zeros(output_dim))

  def load_quantized_weight(self, weight, weight_scaler):
    '''
    Weight shape: [output_channel, input_channel]
    Weight scaler shape: [output_channel]
    '''
    if self.blocksize == -1:
      # Per-channel quant.
      _check_per_channel_quant_weight_dtype_shapes(self.input_dim,
                                                   self.output_dim, weight,
                                                   weight_scaler)
      self.weight = weight
      self.weight_scaler = weight_scaler
    else:
      assert False, "Only per-channel quantization is supported."

  def forward(self, x):
    if self.blocksize == -1:
      return torch.ops.xla.quantized_matmul(x, self.weight, self.weight_scaler)
    else:
      assert False, "Only per-channel quantization is supported."
