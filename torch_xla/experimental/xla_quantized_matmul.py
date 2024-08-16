import torch
import torch.nn.functional as F
import torch_xla
from torch.library import impl
from torch_xla.core.xla_model import XLA_LIB

XLA_LIB.define(
    "quantized_matmul(Tensor x, Tensor w, Tensor scale, Tensor? zero_point=None, int? block_size=-1, bool? int4_weight=False, bool? quantize_activation=False) -> Tensor"
)


def _check_per_channel_quant_weight_dtype_shapes(input_dim, output_dim, w,
                                                 w_scaler, zero_point):
  assert w.dtype == torch.int8, f"Weight dtype is expected to be torch.int8, got {w.dtype}."
  assert w.dim(
  ) == 2, f"Weight tensor is expected to be 2D, got {w.dim()}D Tensor."
  w_shape = list(w.shape)
  assert output_dim == w_shape[0] and input_dim == w_shape[
      1], f"Weight shape is expected to be [output_dim, input_dim], output_dim: {output_dim}, input_dim: {input_dim}, but got {w_shape}."
  assert w_scaler.dim() == 1 and w_scaler.shape[0] == w_shape[
      0], f"weight scaler shape is expect to be [out_channel,], got {w_scaler.shape}, weight shape {w_shape}."
  if zero_point is not None:
    assert zero_point.dim() == 1 and w_scaler.shape[0] == w_shape[
        0], f"zero point shape is expect to be [out_channel,], got {zero_point.shape}, weight shape {w_shape}."


def _check_blockwise_quant_weight_dtype_shapes(input_dim, output_dim,
                                               block_size, w, w_scaler,
                                               zero_point):
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
  if zero_point is not None:
    assert zero_point.dim() == 2, (
        f"zero_point is expected to be 2D, got {zero_point.dim()}D Tensor.")
    assert zero_point.shape[0] == w_shape[0] and zero_point.shape[1] == w_shape[
        -1], (
            f"zero_point shape is expect to be [in_channel / block_size, out_channel], "
            f"got {zero_point.shape}, weight shape {w_shape}.")


def _quantize_tensor(x: torch.Tensor, n_bits: int = 8, dim: int = -1):
  """
  Quantizes a tensor to a lower bit representation.

  Args:
    x (torch.Tensor): The input tensor to be quantized.
    n_bits (int, optional): The number of bits to represent the quantized tensor. Defaults to 8.
    dim (int, optional): The dimension along which to compute the maximum value for scaling. Defaults to -1.

  Returns:
    torch.Tensor: The quantized tensor. (In int8 container)
    torch.Tensor: The scaling factor used for quantization. (Same dtype as x)
  """
  max_val = torch.amax(torch.abs(x), dim=dim, keepdim=True)
  int_min = -2**(n_bits - 1)
  int_max = 2**(n_bits - 1) - 1
  scale = max_val / int_max
  x_int = torch.clamp(torch.round(x / scale), int_min, int_max).to(torch.int8)
  return x_int, scale.to(x.dtype)


@impl(XLA_LIB, "quantized_matmul", "XLA")
def quantized_matmul_xla(x: torch.Tensor,
                         w: torch.Tensor,
                         scaler: torch.Tensor,
                         zero_point: torch.Tensor = None,
                         block_size: int = -1,
                         int4_weight: bool = False,
                         quantize_activation: bool = False):
  """Quantized Matrix Multiply op on XLA devices.

  Args:
      x: torch.Tensor - Activation of Matmul [..., in_channel].
      w: torch.Tensor - Weight Tensor.
         per-channel quant: torch.int8 x [out_channel, in_channel].
         block_wise quant: torch.int8 x [in_channel / block_size, block_size, out_channel].
      scaler: torch.Tensor - Weight scaler.
         per-channel quant: [out_channel,].
         blockwise quant: [in_channel / block_size, out_channel].
      zero_point: Optional[torch.Tensor] - Zero point tensor.
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
                                                 w, scaler, zero_point)
    if quantize_activation:
      x, x_scale = _quantize_tensor(x)
      out = torch_xla._XLAC._xla_dot_general(
          x, w, (([-1], [-1]), ()), preferred_element_type=torch.int32)
    else:
      out = F.linear(x, w)
    out = out * scaler
  else:
    # Blockwise quant.
    assert quantize_activation == False, (
        "Blockwise quantization does not support activation quantization.")
    _check_blockwise_quant_weight_dtype_shapes(x.shape[-1], w.shape[-1],
                                               block_size, w, scaler,
                                               zero_point)
    x = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    out = torch.einsum('scn,...sc->...sn', w, x)
    out = torch.einsum('sn,...sn->...n', scaler, out)
  if zero_point is not None:
    if block_size == -1:
      # Per-channel quant.
      zp_out = torch.einsum("...c,z->...z", x, zero_point)
    else:
      # Blockwise quant.
      zp_out = x.sum(dim=-1)
      zp_out = torch.matmul(zp_out, zero_point)
    out -= zp_out
  if quantize_activation:
    out = out * x_scale
  return out


@impl(XLA_LIB, "quantized_matmul", "CompositeExplicitAutograd")
def quantized_matmul(x: torch.Tensor,
                     w: torch.Tensor,
                     scaler: torch.Tensor,
                     zero_point: torch.Tensor = None,
                     block_size: int = -1,
                     int4_weight: bool = False,
                     quantize_activation: bool = False):
  if block_size == -1:
    # Per-channel quant.
    _check_per_channel_quant_weight_dtype_shapes(x.shape[-1], scaler.shape[0],
                                                 w, scaler, zero_point)
    w = w.to(x.dtype)
    if quantize_activation:
      x, x_scale = _quantize_tensor(x)
      # Upcast to torch.int32 to make sure the mamtul output is int32.
      # Otherwise it will be int8 which causes overflow.
      x = x.to(torch.int32)
      w = w.to(torch.int32)
    out = F.linear(x, w)
    out = out * scaler
  else:
    # Blockwise quant.
    assert quantize_activation == False, (
        "Blockwise quantization does not support activation quantization.")
    _check_blockwise_quant_weight_dtype_shapes(x.shape[-1], w.shape[-1],
                                               block_size, w, scaler,
                                               zero_point)
    x = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    w = w.to(x.dtype)
    out = torch.einsum('scn,...sc->...sn', w, x)
    out = torch.einsum('sn,...sn->...n', scaler, out)

  if zero_point is not None:
    if block_size == -1:
      # Per-channel quant.
      zp_out = torch.einsum("...c,z->...z", x, zero_point)
    else:
      # Blockwise quant.
      zp_out = x.sum(dim=-1)
      zp_out = torch.matmul(zp_out, zero_point)
    out -= zp_out
  if quantize_activation:
    out = out * x_scale
  return out


class XlaQuantizedLinear(torch.nn.Module):

  def __init__(self,
               input_dim,
               output_dim,
               is_symmetric: bool = False,
               block_size: int = -1,
               int4_weight: bool = False,
               quantize_activation: bool = False):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.is_symmetric = is_symmetric
    self.block_size = block_size
    self.int4_weight = int4_weight
    self.quantize_activation = quantize_activation
    self.register_buffer('weight',
                         torch.zeros(output_dim, input_dim).to(torch.int8))
    self.register_buffer('weight_scaler', torch.zeros(output_dim))
    if not self.is_symmetric:
      self.register_buffer('zero_point', torch.zeros(output_dim))
    else:
      self.zero_point = None

  def load_quantized_weight(self, weight, weight_scaler, zero_point=None):
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
                                                   weight_scaler, zero_point)
    else:
      # Blockwise quant.
      _check_blockwise_quant_weight_dtype_shapes(self.input_dim,
                                                 self.output_dim,
                                                 self.block_size, weight,
                                                 weight_scaler, zero_point)
    self.weight = weight
    self.weight_scaler = weight_scaler
    self.zero_point = zero_point

  def forward(self, x):
    return torch.ops.xla.quantized_matmul(
        x,
        self.weight,
        self.weight_scaler,
        self.zero_point,
        block_size=self.block_size,
        int4_weight=self.int4_weight,
        quantize_activation=self.quantize_activation)
