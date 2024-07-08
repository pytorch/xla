import torch
import torch.nn.functional as F
import torch_xla
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "quantized_matmul(Tensor x, Tensor w, Tensor scale, int? block_size=-1, bool? int4_weight=False, bool? quantize_activation=False) -> Tensor"
)


def _check_per_channel_quant_weight_dtype_shapes(input_dim, output_dim, w,
                                                 w_scaler):
  assert w.dtype == torch.int8, f"Weight dtype is expected to be torch.int8, got {w.dtype}."
  assert w.dim(
  ) == 2, f"Weight tensor is expected to be 2D, got {w.dim()}D Tensor."
  w_shape = list(w.shape)
  assert output_dim == w_shape[0] and input_dim == w_shape[
      1], f"Weight shape is expected to be [output_dim, input_dim], output_dim: {output_dim}, input_dim: {input_dim}, but got {w_shape}."
  assert w_scaler.dim() == 1 and w_scaler.shape[0] == w_shape[
      0], f"weight scaler shape is expect to be [out_channel,], got {w_scaler.shape}, weight shape {w_shape}."


def _check_blockwise_quant_weight_dtype_shapes(input_dim, output_dim,
                                               block_size, w, w_scaler):
  assert w.dtype == torch.int8, (
      f"Weight dtype is expected to be torch.int8, got {w.dtype}.")
  assert w.dim() == 3, (
      f"Weight tensor is expected to be 3D, got {w.dim()}D Tensor.")
  w_shape = list(w.shape)
  assert input_dim % block_size == 0, (
      f"input_dim should be divisible by block_size, "
      f"got input_dim: {input_dim}, block_size: {block_size}.")
  assert w_shape[0] == input_dim / block_size and w_shape[1] == block_size, (
      f"Weight shape is expected to be [input_dim / block_size, block_size, output_dim], "
      f"input_dim: {input_dim}, block_size: {block_size}, output_dim: {output_dim}, "
      f"but got {w_shape}.")
  assert w_scaler.dim() == 2, (
      f"weight scaler is expected to be 2D, got {w_scaler.dim()}D Tensor.")
  assert w_scaler.shape[0] == w_shape[0] and w_scaler.shape[1] == w_shape[-1], (
      f"weight scaler shape is expect to be [in_channel / block_size, out_channel], "
      f"got {w_scaler.shape}, weight shape {w_shape}.")


@impl(XLA_LIB, "quantized_matmul", "XLA")
def quantized_matmul_xla(x: torch.Tensor,
                         w: torch.Tensor,
                         scaler: torch.Tensor,
                         block_size: int = -1,
                         int4_weight: bool = False):
  """Quantized Matrix Multiply op on XLA devices.

  Args:
      x: torch.Tensor - Activation of Matmul [..., in_channel].
      w: torch.Tensor - Weight Tensor.
         per-channel quant: torch.int8 x [out_channel, in_channel].
         block_wise quant: torch.int8 x [in_channel / block_size, block_size, out_channel].
      scaler: torch.Tensor - Weight scaler.
         per-channel quant: [out_channel,].
         blockwise quant: [in_channel / block_size, out_channel].
      block_size: The blocksize for blockwise quantization, -1 for per-channel quantization.
      int4_weight: if the weights are int4, the int4 weights need to be stored in a int8
                   container (unpacked).
  """
  if int4_weight:
    # Reinterpret cast the weight to s4 dtype in XLA.
    w = torch_xla._XLAC._xla_cast_int4(w, w.cpu().flatten().numpy().tolist())
  if block_size == -1:
    # Per-channel quant.
    _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0],
                                                 w, scaler)
    return F.linear(x, w) * scaler
  else:
    # Blockwise quant.
    _check_blockwise_quant_weight_dtype_shapes(x.shape[-1], w.shape[-1],
                                               block_size, w, scaler)
    x = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    out = torch.einsum('scn,...sc->...sn', w, x)
    out = torch.einsum('sn,...sn->...n', scaler, out)
    return out


@impl(XLA_LIB, "quantized_matmul", "CompositeExplicitAutograd")
def quantized_matmul(x: torch.Tensor,
                     w: torch.Tensor,
                     scaler: torch.Tensor,
                     block_size: int = -1,
                     int4_weight: bool = False):
  if block_size == -1:
    # Per-channel quant.
    _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0],
                                                 w, scaler)
    w = w.to(x.dtype)
    return torch.mul(F.linear(x, w), scaler)
  else:
    # Blockwise quant.
    _check_blockwise_quant_weight_dtype_shapes(x.shape[-1], w.shape[-1],
                                               block_size, w, scaler)
    x = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    w = w.to(x.dtype)
    out = torch.einsum('scn,...sc->...sn', w, x)
    out = torch.einsum('sn,...sn->...n', scaler, out)
    return out


class XlaQuantizedLinear(torch.nn.Module):

  def __init__(self,
               input_dim,
               output_dim,
               block_size=-1,
               int4_weight: bool = False):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.block_size = block_size
    self.int4_weight = int4_weight
    self.register_buffer('weight',
                         torch.zeros(output_dim, input_dim).to(torch.int8))
    self.register_buffer('weight_scaler', torch.zeros(output_dim))

  def load_quantized_weight(self, weight, weight_scaler):
    '''
    weight (Tensor):
      per-channel quant: [out_channel, in_channel].
      block_wise quant: [in_channel / block_size, block_size, out_channel].

    weight_scaler (Tensor):
      per-channel quant: [out_channel,].
      blockwise quant: [in_channel / block_size, out_channel].
    '''
    if self.block_size == -1:
      # Per-channel quant.
      _check_per_channel_quant_weight_dtype_shapes(self.input_dim,
                                                   self.output_dim, weight,
                                                   weight_scaler)
    else:
      # Blockwise quant.
      _check_blockwise_quant_weight_dtype_shapes(self.input_dim,
                                                 self.output_dim,
                                                 self.block_size, weight,
                                                 weight_scaler)
    self.weight = weight
    self.weight_scaler = weight_scaler

  def forward(self, x):
    return torch.ops.xla.quantized_matmul(
        x,
        self.weight,
        self.weight_scaler,
        block_size=self.block_size,
        int4_weight=self.int4_weight)
