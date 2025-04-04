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

  def build_models(self, input_size, hidden_size, num_layers, bias):
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

    # Copy parameters from the upstream GRU to our scan-based GRU.
    # This ensures that the scan-based GRU has the same parameters as the
    # upstream GRU and both models are parameterized the same way.
    scan_gru.load_state_dict(gru.state_dict(), strict=True)

    return gru, scan_gru

  def check_gradients(self,
                      inp1,
                      hx1,
                      inp2,
                      hx2,
                      num_layers,
                      gru,
                      scan_gru,
                      atol=None,
                      rtol=None):
    # Compare gradients for input and initial hidden state.
    assert inp1.grad is not None
    assert hx1.grad is not None
    assert inp2.grad is not None
    assert hx2.grad is not None
    torch.testing.assert_close(
        inp1.grad,
        inp2.grad,
        msg=lambda msg: f"Input gradient mismatch. {msg}",
        check_device=False,
        atol=atol,
        rtol=rtol)
    torch.testing.assert_close(
        hx1.grad,
        hx2.grad,
        msg=lambda msg: f"Hidden state gradient mismatch. {msg}",
        check_device=False,
        atol=atol,
        rtol=rtol)

    # Compare gradients for all parameters.
    params_to_check = ['weight_ih', 'weight_hh']
    assert scan_gru.bias == gru.bias
    if scan_gru.bias:
      params_to_check += ['bias_ih', 'bias_hh']
    for layer in range(num_layers):
      for name in params_to_check:
        param1 = getattr(gru, f'{name}_l{layer}')
        param2 = getattr(scan_gru, f'{name}_l{layer}')
        torch.testing.assert_close(
            param1.grad,
            param2.grad,
            msg=lambda msg:
            f"Gradient mismatch in {name} at layer {layer}. {msg}",
            check_device=False,
            atol=atol,
            rtol=rtol)

  def test_scan_gru_and_upstream_gru_parameter_independency(self):
    """
    Ensures that the parameters of the scan-based GRU and upstream GRU are independent even the parameters of the scan-based GRU are initialized using the upstream GRU.
    """
    input_size, hidden_size, num_layers = 16, 32, 2
    gru, scan_gru = self.build_models(input_size, hidden_size, num_layers, True)
    gru = gru.cpu()
    scan_gru = scan_gru.to('xla')
    torch_xla.sync()

    with torch.no_grad():
      gru_weight_ih_l0 = gru.state_dict()['weight_ih_l0']
      scan_gru_weight_ih_l0 = scan_gru.state_dict()['weight_ih_l0']

      # Compare the parameters of the GRU and scan-based GRU before changing.
      torch.testing.assert_close(
          gru_weight_ih_l0,
          scan_gru_weight_ih_l0,
          msg=lambda msg: f"weight_ih_l0 mismatch. {msg}",
          check_device=False)

      # Change the parameters of the GRU with random numbers.
      gru_weight_ih_l0.uniform_(-1, 1)

      # Assert not close after the change.
      try:
        torch.testing.assert_close(
            gru_weight_ih_l0,
            scan_gru_weight_ih_l0,
            msg=lambda msg: f"weight_ih_l0 mismatch. {msg}",
            check_device=False)
        raise AssertionError(
            "weight_ih_l0 should not be close after changing the GRU parameters."
        )
      except AssertionError as e:
        if str(e).startswith("weight_ih_l0 mismatch."):
          pass
        else:
          raise e

  @parameterized.parameters(True, False)
  def test_scan_gru_vs_pytorch_xla_for_loop(self, bias):
    """
    Compare scan-based GRU against upstream GRU both compiled with XLA.
    """
    seq_len, batch_size, input_size, hidden_size, num_layers = 16, 4, 16, 32, 2
    gru, scan_gru = self.build_models(input_size, hidden_size, num_layers, bias)
    gru, scan_gru = gru.to('xla'), scan_gru.to('xla')
    torch_xla.sync()

    # Prepare input and initial hidden states.
    inp1 = torch.randn(seq_len, batch_size,
                       input_size).to('xla').requires_grad_(True)
    inp2 = inp1.clone().detach().requires_grad_(True)
    hx1 = torch.randn(num_layers, batch_size,
                      hidden_size).to('xla').requires_grad_(True)
    hx2 = hx1.clone().detach().requires_grad_(True)
    torch_xla.sync()

    # Forward passes.
    out1, h1 = gru(inp1, hx1)
    torch_xla.sync()

    out2, h2 = scan_gru(inp2, hx2)
    torch_xla.sync()

    # Compare the numerical outputs.
    torch.testing.assert_close(out1, out2, check_device=False)
    torch.testing.assert_close(h1, h2, check_device=False)

    # Compute losses.
    loss1 = out1.sum() + h1.sum()
    loss2 = out2.sum() + h2.sum()

    # Backward passes.
    loss1.backward()
    loss2.backward()
    torch_xla.sync()

    self.check_gradients(inp1, hx1, inp2, hx2, num_layers, gru, scan_gru)

  @parameterized.parameters(True, False)
  def test_scan_gru_vs_pytorch_native_cpu(self, bias):
    """
    Compare scan-based GRU compiled with XLA against upstream GRU run with PyTorch eager.
    """
    seq_len, batch_size, input_size, hidden_size, num_layers = 2048, 4, 16, 32, 5
    gru, scan_gru = self.build_models(input_size, hidden_size, num_layers, bias)
    gru = gru.cpu()
    scan_gru = scan_gru.to('xla')
    torch_xla.sync()

    # Prepare input and initial hidden states.
    inp1 = torch.randn(seq_len, batch_size, input_size).requires_grad_(True)
    inp2 = inp1.to('xla').clone().detach().requires_grad_(True)
    hx1 = torch.randn(num_layers, batch_size, hidden_size).requires_grad_(True)
    hx2 = hx1.to('xla').clone().detach().requires_grad_(True)
    torch_xla.sync()

    # Forward passes.
    out1, h1 = gru(inp1, hx1)
    torch_xla.sync()

    out2, h2 = scan_gru(inp2, hx2)
    torch_xla.sync()

    # Compare the numerical outputs.
    torch.testing.assert_close(
        out1, out2, check_device=False, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(h1, h2, check_device=False, atol=1e-3, rtol=1e-3)

    # Compute losses.
    loss1 = out1.sum() + h1.sum()
    loss2 = out2.sum() + h2.sum()

    # Backward passes.
    loss1.backward()
    loss2.backward()
    torch_xla.sync()

    # Gradient thresholds are relaxed because numerical differences between TPU
    # and CPU adds up to a non-trivial impact over 2048 steps.
    self.check_gradients(
        inp1, hx1, inp2, hx2, num_layers, gru, scan_gru, atol=0.05, rtol=0.05)


if __name__ == "__main__":
  torch_xla._XLAC._xla_set_mat_mul_precision('highest')
  absltest.main()
