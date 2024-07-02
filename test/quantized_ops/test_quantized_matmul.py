import re
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_quantized_matmul
from torch_xla import runtime as xr
from torch_xla.experimental.xla_quantized_matmul import XlaQuantizedLinear
from torch.ao.quantization.utils import determine_qparams

torch.manual_seed(123456)

device = xm.xla_device()


class M(torch.nn.Module):

  def __init__(self, input_dim, output_dim):
    super(M, self).__init__()
    # Define a linear layer
    self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

  def weight_quantization_rtn(self,
                              linear,
                              n_bits=8,
                              block_size=-1,
                              quant_method=torch.per_channel_symmetric):
    '''
    Quantize linear weight using Round-To-Nearest(RTN) algorithm.
    '''
    assert isinstance(self.linear, torch.nn.Linear)
    w_fp = linear.weight.data
    if block_size == -1:
      min_val, max_val = torch.aminmax(
          w_fp, dim=1)  # min_val, max_val [out_dim]
      int_min = -2**(n_bits - 1)
      int_max = 2**(n_bits - 1) - 1
      scaler, zero_point = determine_qparams(
          min_val,
          max_val,
          int_min,
          int_max,
          dtype=torch.int8,
          eps=torch.Tensor([1e-5]),
          has_customized_qrange=False,
          qscheme=quant_method)
      w_int = torch.ops.quantized_decomposed.quantize_per_channel(
          w_fp, scaler, zero_point, 0, int_min, int_max, torch.int8)
      return w_int, scaler.to(w_fp.dtype), zero_point
    else:
      assert w_fp.shape[1] % block_size == 0
      output_dim = w_fp.shape[0]
      input_dim = w_fp.shape[1]
      w_fp = w_fp.reshape(output_dim * input_dim // block_size, block_size)
      min_val, max_val = torch.aminmax(
          w_fp, dim=1)  # min_val, max_val [out_dim]
      int_min = -2**(n_bits - 1)
      int_max = 2**(n_bits - 1) - 1
      scaler, zero_point = determine_qparams(
          min_val,
          max_val,
          int_min,
          int_max,
          dtype=torch.int8,
          eps=torch.Tensor([1e-5]),
          has_customized_qrange=False,
          qscheme=quant_method)
      w_int = torch.ops.quantized_decomposed.quantize_per_channel(
          w_fp, scaler, zero_point, 0, int_min, int_max, torch.int8)
      w_int = w_int.reshape(output_dim, input_dim // block_size,
                            block_size).permute(1, 2, 0)
      scaler = scaler.to(w_fp.dtype).reshape(output_dim,
                                             input_dim // block_size).permute(
                                                 1, 0)
      return w_int, scaler, zero_point

  def replace_with_xla_quantized_matmul(self, n_bit=8, block_size=-1):
    assert isinstance(self.linear, torch.nn.Linear)
    w_int, scaler, _ = self.weight_quantization_rtn(
        self.linear, n_bits=n_bit, block_size=block_size)
    use_int4_weight = n_bit == 4
    q_linear = XlaQuantizedLinear(
        self.linear.in_features,
        self.linear.out_features,
        block_size=block_size,
        int4_weight=use_int4_weight)
    q_linear.load_quantized_weight(w_int, scaler)
    self.linear = q_linear

  def forward(self, x):
    # Forward pass through the linear layer
    return self.linear(x)


class QuantizedTest(unittest.TestCase):

  def _calc_cosine_dist(self, x, y):
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)
    return (torch.dot(x, y) / (x.norm() * y.norm())).item()

  def test_q_linear_module_per_channel(self):

    with torch.no_grad():
      m = M(5, 8)
      x = torch.randn(3, 5)
      out_fp = m(x)
      m.replace_with_xla_quantized_matmul()
      out_quant = m(x)

      m = m.to(device)
      x = x.to(device)
      out_quant_xla = m(x)
      self.assertTrue(torch.allclose(out_fp, out_quant, atol=0.01))
      self.assertTrue(torch.allclose(out_quant_xla.cpu(), out_quant))

  def test_q_linear_module_dynamo(self):

    with torch.no_grad():
      m = M(5, 8)
      x = torch.randn(3, 5)
      out_fp = m(x)
      m.replace_with_xla_quantized_matmul()
      out_quant = m(x)
      m = m.to(device)
      m_dynamo = torch.compile(m, backend="openxla")
      out_quant_dynamo = m_dynamo(x.to(device))
      self.assertTrue(torch.allclose(out_fp, out_quant, atol=0.01))
      self.assertTrue(torch.allclose(out_quant_dynamo.cpu(), out_quant))

  def test_q_linear_hlo(self):
    with torch.no_grad():
      x = torch.randn((3, 5), dtype=torch.bfloat16).to(device)
      w_int = torch.randint(-128, 127, (8, 5), dtype=torch.int8).to(device)
      scaler = torch.randn((8,), dtype=torch.bfloat16).to(device)

      output = torch.ops.xla.quantized_matmul(x, w_int, scaler)
      hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
      self.assertTrue(re.search(r'bf16.*dot.*bf16.*s8', hlo) is not None)

  def test_int4_per_channel_matmul(self):
    weight = torch.randint(-8, 7, (4, 2)).to(torch.int8)
    weight_scaler = torch.randn(4).to(torch.bfloat16)
    x = torch.ones(3, 2).to(torch.bfloat16)

    torch_out = torch.ops.xla.quantized_matmul(x, weight, weight_scaler)

    x = x.to(device)
    weight = weight.to(device)
    weight_scaler = weight_scaler.to(device)

    xla_out = torch.ops.xla.quantized_matmul(
        x, weight, weight_scaler, int4_weight=True)

    hlo = torch_xla._XLAC._get_xla_tensors_hlo([xla_out])
    self.assertTrue(re.search(r'bf16.*dot.*bf16.*s4', hlo) is not None)

    # Dot with int4 weight is only supported on TPU
    if xr.device_type() == 'TPU':
      self.assertGreater(
          self._calc_cosine_dist(xla_out.cpu(), torch_out), 0.999999)

  def test_int4_per_channel_linear_module(self):
    m = M(5, 8)
    x = torch.randn(3, 5)
    out_fp = m(x)
    m.replace_with_xla_quantized_matmul(n_bit=4)
    out_quant = m(x)
    self.assertGreater(self._calc_cosine_dist(out_fp, out_quant), 0.99)

    # Dot with int4 weight is only supported on TPU
    if xr.device_type() == 'TPU':
      m = m.to(device)
      x = x.to(device)
      out_quant_xla = m(x)
      self.assertGreater(
          self._calc_cosine_dist(out_quant_xla.cpu(), out_quant), 0.999999)

  def test_blockwise_matmul_op(self):
    input_features = 6
    out_features = 8
    block_size = 2
    batch_size = 3
    for n_bit in [4]:
      with self.subTest(n_bit=n_bit):
        weight = torch.randint(-8, 7, (input_features // block_size, block_size,
                                       out_features)).to(torch.int8)
        weight_scaler = torch.ones(input_features // block_size, out_features)
        x = torch.rand(batch_size, input_features)

        # Fake quantize output.
        w_dq = (weight * weight_scaler.unsqueeze(1)).reshape(
            input_features, out_features)
        fake_quant_out = torch.matmul(x, w_dq)
        # Eager output.
        torch_out = torch.ops.xla.quantized_matmul(
            x, weight, weight_scaler, block_size=block_size)
        self.assertGreater(
            self._calc_cosine_dist(fake_quant_out, torch_out), 0.99999)
        # XLA Output.
        if not (n_bit == 4 and xr.device_type() != 'TPU'):
          x = x.to(device)
          weight = weight.to(device)
          weight_scaler = weight_scaler.to(device)
          xla_out = torch.ops.xla.quantized_matmul(
              x, weight, weight_scaler, block_size=block_size)
          self.assertTrue(torch.allclose(torch_out, xla_out.cpu(), atol=0.03))

  def test_blockwise_linear_module(self):
    for n_bit in [4, 8]:
      with self.subTest(n_bit=n_bit):
        m = M(6, 8)
        x = torch.randn(3, 6)
        out_fp = m(x)
        m.replace_with_xla_quantized_matmul(n_bit=8, block_size=2)
        out_quant = m(x)
        self.assertGreater(self._calc_cosine_dist(out_fp, out_quant), 0.99)

        # Dot with int4 weight is only supported on TPU
        if not (n_bit == 4 and xr.device_type() != 'TPU'):
          m = m.to(device)
          x = x.to(device)
          out_quant_xla = m(x)
          self.assertGreater(
              self._calc_cosine_dist(out_quant_xla.cpu(), out_quant), 0.999999)


if __name__ == '__main__':
  unittest.main()
