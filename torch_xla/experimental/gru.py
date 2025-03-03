import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_xla.experimental.scan import scan


class GRU(nn.Module):
  r"""
  PyTorch/XLA GRU implemented using scan.

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

  def __init__(self,
               input_size,
               hidden_size,
               num_layers=1,
               bias=True,
               dropout=0.0):
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
      assert hx.size(0) == self.num_layers, \
        "Mismatch in number of layers for hidden state."

    # The output of one layer is the input to the next.
    output = input
    hidden_states = []

    # Loop over each layer.
    for layer in range(self.num_layers):
      init = {
          'h': hx[layer],
          'w_ih': self.weight_ih[layer],
          'w_hh': self.weight_hh[layer]
      }
      if self.bias:
        init['b_ih'] = self.bias_ih[layer]
        init['b_hh'] = self.bias_hh[layer]

      # Define the step function for scanning over time.
      # x_t: (batch, current_input_size)
      # h: (batch, hidden_size)
      # carry: dictionary containing h and weights/biases.
      def step_fn(carry, x_t):
        h = carry['h']
        w_ih = carry['w_ih']
        w_hh = carry['w_hh']
        b_ih = carry.get('b_ih')
        b_hh = carry.get('b_hh')

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

        carry_new = {
            'h': h_new,
            'w_ih': w_ih,
            'w_hh': w_hh,
        }
        if b_ih is not None:
          carry_new['b_ih'] = b_ih
        if b_hh is not None:
          carry_new['b_hh'] = b_hh
        return carry_new, h_new

      # Use scan to iterate over the time dimension.
      # Here, scan(fn, init, xs) applies step_fn to each time slice of `output`.
      final_carry, layer_output = scan(fn=step_fn, init=init, xs=output)
      hidden_states.append(final_carry['h'])
      # Apply dropout on the output of the current layer (if not the final layer).
      if layer < self.num_layers - 1 and self.dropout > 0:
        layer_output = F.dropout(
            layer_output, p=self.dropout, training=self.training)
      output = layer_output
    assert output.size(0) == seq_len

    # Stack the final hidden states for each layer.
    hidden = torch.stack(hidden_states, dim=0)
    return output, hidden
