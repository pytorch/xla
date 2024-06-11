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
                              quant_method=torch.per_channel_symmetric):
    '''
    Quantize linear weight using Round-To-Nearest(RTN) algorithm.
    '''
    assert isinstance(self.linear, torch.nn.Linear)
    w_fp = linear.weight.data
    min_val, max_val = torch.aminmax(w_fp, dim=1)  # min_val, max_val [out_dim]
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

  def replace_with_xla_quantized_matmul(self, n_bit=8):
    assert isinstance(self.linear, torch.nn.Linear)
    w_int, scaler, _ = self.weight_quantization_rtn(self.linear, n_bit)
    use_int4_weight = n_bit == 4
    q_linear = XlaQuantizedLinear(
        self.linear.in_features,
        self.linear.out_features,
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


if __name__ == '__main__':
  unittest.main()
