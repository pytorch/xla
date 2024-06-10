import torch
import torch.nn.functional as F
import torch_xla
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "quantized_matmul(Tensor x, Tensor w, Tensor scale, int? blocksize=-1, bool? int4_weight=False, bool? quantize_activation=False) -> Tensor"
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


def pack_4bit(x, dtype=torch.int8):
  # Not used now, we don't do packed weight because reinterpret_cast
  # doesn't work on TPU.
  # Pack 4 bit tensor into (u)int8 or (u)int32
  # Always pack along the last dim
  assert dtype == torch.int8, "Only int8 pack/unpack is supported now."
  num_int4_per_element = 2 if (dtype == torch.int8 or
                               dtype == torch.uint8) else 8
  assert x.shape[-1] % num_int4_per_element == 0
  packed = x.reshape(*x.shape[:-1], x.shape[-1] // num_int4_per_element,
                     num_int4_per_element)
  shift = torch.arange(num_int4_per_element).to(packed.dtype)
  shift *= 4
  # Clear higher bits.
  packed = packed.bitwise_and(torch.tensor([0x0F])).to(torch.uint8)
  packed = torch.bitwise_left_shift(packed, shift).to(torch.uint8)
  packed = torch.sum(packed.view(torch.uint8), -1).to(torch.uint8)
  packed = packed.view(x.dtype)
  return packed


def unpack_4bit(x, dtype=torch.int8):
  # Not used now, we don't do packed weight because reinterpret_cast
  # doesn't work on TPU.
  # Unpack 4 bit tensor into int8 or int32
  # Always unpack along the last dim
  assert dtype == torch.int8, "Only int8 pack/unpack is supported now."
  num_int4_per_element = 2 if (dtype == torch.int8 or
                               dtype == torch.uint8) else 8
  unpacked = x.unsqueeze(-1).expand(*x.shape, 2)
  shift = torch.arange(num_int4_per_element).flip(0).to(unpacked.dtype)
  shift *= 4
  # Left shift first then right shift to preserve the sign bit in int8.
  unpacked = unpacked.bitwise_left_shift(shift).bitwise_right_shift(
      num_int4_per_element * 4 - 4)
  unpacked = unpacked.flatten(-2)
  unpacked = unpacked.view(dtype)
  return unpacked


@impl(XLA_LIB, "quantized_matmul", "XLA")
def quantized_matmul_xla(x: torch.Tensor,
                         w: torch.Tensor,
                         scaler: torch.Tensor,
                         blocksize: int = -1,
                         int4_weight: bool = False):
  """Quantized Matrix Multiply op on XLA devices.

  Args:
      x: torch.Tensor - Activation of Matmul [..., in_channel].
      w: torch.Tensor - Weight Tensor.
         per-channel quant: torch.int8 x [out_channel, in_channel].
      scaler: torch.Tensor - Weight scaler.
         per-channel quant: [out_channel,].
      blocksize: blocksize for blockwise quantization, -1 for per-channel quantization.
      int4_weight: if the weights are int4, the int4 weights need to be stored in a int8
                   container (unpacked).
  """
  assert blocksize == -1, "blockwise quantization is not supported yet."
  if int4_weight:
    # Reinterpret cast the weight to s4 dtype in XLA.
    w = torch_xla._XLAC._xla_reinterpret_cast_int4(
        w,
        w.cpu().flatten().numpy().tolist())
  # Per-channel quant.
  _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0], w,
                                               scaler)
  return F.linear(x, w) * scaler


@impl(XLA_LIB, "quantized_matmul", "CompositeExplicitAutograd")
def quantized_matmul(x: torch.Tensor,
                     w: torch.Tensor,
                     scaler: torch.Tensor,
                     blocksize: int = -1,
                     int4_weight: bool = False):
  assert blocksize == -1, "blockwise quantization is not supported yet."
  # Per-channel quant.
  _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0], w,
                                               scaler)
  w = w.to(x.dtype)
  return torch.mul(F.linear(x, w), scaler)


class XlaQuantizedLinear(torch.nn.Module):

  def __init__(self,
               input_dim,
               output_dim,
               blocksize=-1,
               int4_weight: bool = False):
    super().__init__()
    assert blocksize == -1, "Only per-channel quantization is supported."
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.blocksize = blocksize
    self.int4_weight = int4_weight
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
      return torch.ops.xla.quantized_matmul(
          x, self.weight, self.weight_scaler, int4_weight=self.int4_weight)
    else:
      assert False, "Only per-channel quantization is supported."
