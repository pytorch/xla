import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla
from torch_xla.experimental.scan import scan


class ScanGRU(nn.Module):

  def __init__(self,
               input_size,
               hidden_size,
               num_layers=1,
               bias=True,
               dropout=0.0):
    r"""
    Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0

    This implementation has the following differences from the GRU module in PyTorch upstream:

    - Only supports unidirectional GRU.
    - Only supports inputs in the `(seq, batch, feature)` format (i.e. `batch_first = False`).

    """
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.dropout = dropout

    # Create parameters for each layer.
    # For layer 0, the input dimension is `input_size`, otherwise it's `hidden_size`.
    self.weight_ih = nn.ParameterList()
    self.weight_hh = nn.ParameterList()
    if bias:
      self.bias_ih = nn.ParameterList()
      self.bias_hh = nn.ParameterList()

    for layer in range(num_layers):
      layer_input_size = input_size if layer == 0 else hidden_size
      # weight_ih: combines weights for reset, update, and new gates.
      w_ih = nn.Parameter(torch.Tensor(3 * hidden_size, layer_input_size))
      w_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
      self.weight_ih.append(w_ih)
      self.weight_hh.append(w_hh)
      if bias:
        b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        b_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.bias_ih.append(b_ih)
        self.bias_hh.append(b_hh)
    self.reset_parameters()

  def reset_parameters(self):
    # Initialize parameters uniformly as in the upstream PyTorch GRU.
    stdv = 1.0 / (self.hidden_size**0.5)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input, hx=None):
    """
    Args:
        input: Tensor of shape (seq_len, batch, input_size)
        hx: Optional initial hidden state of shape (num_layers, batch, hidden_size).
            If not provided, defaults to zeros.
    Returns:
        output: Tensor of shape (seq_len, batch, hidden_size) from the last GRU layer.
        hidden: Tensor of shape (num_layers, batch, hidden_size) containing the final hidden state per layer.
    """
    seq_len, batch_size, _ = input.size()
    if hx is None:
      hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
    else:
      assert hx.size(
          0
      ) == self.num_layers, "Mismatch in number of layers for hidden state."

    # The output of one layer is the input to the next.
    output = input
    hidden_states = []

    # Loop over each layer.
    for layer in range(self.num_layers):
      w_ih = self.weight_ih[layer]
      w_hh = self.weight_hh[layer]
      if self.bias:
        b_ih = self.bias_ih[layer]
        b_hh = self.bias_hh[layer]
      else:
        b_ih = b_hh = None

      # Define the step function for scanning over time.
      # x_t: (batch, current_input_size)
      # h: (batch, hidden_size)
      def step_fn(carry, x_t):
        h, w_ih, w_hh, b_ih, b_hh = carry

        # Get input projections
        x_linear = F.linear(x_t, w_ih, b_ih)
        x_r, x_z, x_n = x_linear.chunk(3, dim=1)

        # Get hidden projections
        h_linear = F.linear(h, w_hh, b_hh)
        h_r, h_z, h_n = h_linear.chunk(3, dim=1)

        # Compute reset and update gates
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)

        # Compute the new gate with proper reset gate application
        n = torch.tanh(x_n + r * h_n)

        # Update hidden state
        h_new = (1 - z) * n + z * h

        carry_new = (h_new, w_ih, w_hh, b_ih, b_hh)
        return carry_new, h_new

      # Use scan to iterate over the time dimension.
      # Here, scan(fn, init, xs) applies step_fn to each time slice of `output`.
      (h_final, _, _, _, _), layer_output = scan(
          fn=step_fn, init=(hx[layer], w_ih, w_hh, b_ih, b_hh), xs=output)
      hidden_states.append(h_final)
      # Apply dropout on the output of the current layer (if not the final layer).
      if layer < self.num_layers - 1 and self.dropout > 0:
        layer_output = F.dropout(
            layer_output, p=self.dropout, training=self.training)
      output = layer_output

    # Stack the final hidden states for each layer.
    hidden = torch.stack(hidden_states, dim=0)
    return output, hidden


# Unit test comparing upstream GRU and our scan-based GRU.
def test_gru_scan_vs_pytorch():
  torch.manual_seed(0)
  torch_xla.manual_seed(0)
  # torch.set_default_dtype(torch.bfloat16)

  # seq_len, batch_size, input_size, hidden_size, num_layers = 512, 4, 16, 32, 8
  seq_len, batch_size, input_size, hidden_size, num_layers = 16, 4, 16, 32, 2

  # Create PyTorch upstream GRU and our scan-based GRU.
  gru = nn.GRU(
      input_size,
      hidden_size,
      num_layers=num_layers,
      bias=True,
      batch_first=False,
      dropout=0.0,
      bidirectional=False)
  scan_gru = ScanGRU(
      input_size, hidden_size, num_layers=num_layers, bias=True, dropout=0.0)

  # Copy parameters from the upstream GRU to our ScanGRU.
  for layer in range(num_layers):
    scan_gru.weight_ih[layer].data.copy_(
        getattr(gru, f'weight_ih_l{layer}').data)
    scan_gru.weight_hh[layer].data.copy_(
        getattr(gru, f'weight_hh_l{layer}').data)
    if gru.bias:
      scan_gru.bias_ih[layer].data.copy_(getattr(gru, f'bias_ih_l{layer}').data)
      scan_gru.bias_hh[layer].data.copy_(getattr(gru, f'bias_hh_l{layer}').data)

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
  for layer in range(num_layers):
    for name in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
      param1 = getattr(gru, f'{name}_l{layer}')
      param2 = getattr(scan_gru, name)[layer]
      torch.testing.assert_close(
          param1.grad,
          param2.grad.cpu(),
          check_device=False,
          msg=lambda msg: f"Gradient mismatch in {name} at layer {layer}. {msg}"
      )
  print("All tests passed!")


if __name__ == "__main__":
  test_gru_scan_vs_pytorch()
