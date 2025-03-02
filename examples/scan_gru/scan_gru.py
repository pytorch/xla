import torch
import torch.nn as nn

import torch_xla
from torch_xla.experimental.gru import GRU

from absl.testing import absltest, parameterized


class TestGRU(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    torch_xla.manual_seed(0)

  @parameterized.parameters(True, False)
  def test_gru_scan_vs_pytorch(self, bias):
    """
    Unit test comparing upstream GRU and our scan-based GRU.
    """

    # seq_len, batch_size, input_size, hidden_size, num_layers = 512, 4, 16, 32, 8
    seq_len, batch_size, input_size, hidden_size, num_layers = 16, 4, 16, 32, 2

    # Create PyTorch upstream GRU and our scan-based GRU.
    gru = nn.GRU(
        input_size,
        hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=False,
        dropout=0.0,
        bidirectional=False)
    scan_gru = GRU(
        input_size, hidden_size, num_layers=num_layers, bias=bias, dropout=0.0)

    # Copy parameters from the upstream GRU to our ScanGRU.
    for layer in range(num_layers):
      scan_gru.weight_ih[layer].data.copy_(
          getattr(gru, f'weight_ih_l{layer}').data)
      scan_gru.weight_hh[layer].data.copy_(
          getattr(gru, f'weight_hh_l{layer}').data)
      if gru.bias:
        scan_gru.bias_ih[layer].data.copy_(
            getattr(gru, f'bias_ih_l{layer}').data)
        scan_gru.bias_hh[layer].data.copy_(
            getattr(gru, f'bias_hh_l{layer}').data)

    scan_gru = scan_gru.to('xla')
    torch_xla.sync()

    # Prepare input and initial hidden states.
    inp1 = torch.randn(seq_len, batch_size, input_size, requires_grad=True)
    inp2 = inp1.clone().detach().to('xla').requires_grad_(True)
    hx1 = torch.randn(num_layers, batch_size, hidden_size, requires_grad=True)
    hx2 = hx1.clone().detach().to('xla').requires_grad_(True)

    # Forward passes.
    gru = gru.to('xla')
    out1, h1 = gru(inp1.to('xla'), hx1.to('xla'))
    torch_xla.sync()
    out1 = out1.cpu()
    h1 = h1.cpu()

    out2, h2 = scan_gru(inp2, hx2)
    torch_xla.sync()

    # Compare the numerical outputs.
    torch.testing.assert_close(out1, out2.cpu())
    torch.testing.assert_close(h1, h2.cpu())

    # Compute losses.
    loss1 = out1.sum() + h1.sum()
    loss2 = out2.sum() + h2.sum()

    # Backward passes.
    loss1.backward()
    loss2.backward()
    torch_xla.sync()

    # Compare gradients for input and initial hidden state.
    assert inp1.grad is not None
    assert hx1.grad is not None
    assert inp2.grad is not None
    assert hx2.grad is not None
    torch.testing.assert_close(
        inp1.grad,
        inp2.grad.cpu(),
        msg=lambda msg: f"Input gradient mismatch. {msg}",
        check_device=False)
    torch.testing.assert_close(
        hx1.grad,
        hx2.grad.cpu(),
        msg=lambda msg: f"Hidden state gradient mismatch. {msg}",
        check_device=False)

    # Compare gradients for all parameters.
    params_to_check = ['weight_ih', 'weight_hh']
    if bias:
      params_to_check += ['bias_ih', 'bias_hh']
    for layer in range(num_layers):
      for name in params_to_check:
        param1 = getattr(gru, f'{name}_l{layer}')
        param2 = getattr(scan_gru, name)[layer]
        torch.testing.assert_close(
            param1.grad,
            param2.grad.cpu(),
            check_device=False,
            msg=lambda msg:
            f"Gradient mismatch in {name} at layer {layer}. {msg}")
    print("All tests passed!")


if __name__ == "__main__":
  absltest.main()
