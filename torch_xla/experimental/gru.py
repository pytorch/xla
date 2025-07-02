import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn.utils.rnn import PackedSequence
from typing import overload

from torch_xla.experimental.scan import scan


class GRU(nn.GRU):
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
      batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
      dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
          GRU layer except the last layer, with dropout probability equal to
          :attr:`dropout`. Default: 0
      bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

  This implementation has the following differences from the GRU module in PyTorch upstream:

  - Only supports unidirectional GRU.
  - Only supports inputs in the `(seq, batch, feature)` format (i.e. `batch_first = False`).

  """

  def __new__(cls, *args, **kwargs):
    if ('bidirectional' in kwargs and kwargs['bidirectional'] == True):
      logging.warning(
          "Scan-based GRU only supports unidirectional GRU. (bidirectional = False) "
          "Scan-based GRU falls back to the default nn.GRU implementation instead."
      )
      if nn.GRU._orig is None:
        # If nn.GRU._orig is None, it means that the original GRU has not been
        # patched yet for some reason. The patching should happen in _patched_functions.py.
        # So we need to call the original GRU constructor here.
        return nn.GRU(*args, **kwargs)
      else:
        # If nn.GRU._orig is not None, it means that the original GRU has been
        # patched already. So we need to call the patched GRU constructor here.
        return nn.GRU._orig(*args, **kwargs)
    return super().__new__(cls)

  @overload
  def __init__(
      self,
      input_size: int,
      hidden_size: int,
      num_layers: int = 1,
      bias: bool = True,
      batch_first: bool = False,
      dropout: float = 0.0,
      bidirectional: bool = False,
  ):
    pass

  def __init__(self, *args, **kwargs):
    assert not kwargs.get('bidirectional', False), \
      "GRU only supports unidirectional GRU."
    assert kwargs.get('proj_size', 0) == 0, \
      "GRU only supports no projection."

    super().__init__(*args, **kwargs)

  def forward(self, input, hx=None):
    """
    Args:
        input: Tensor of shape (seq_len, batch, input_size)
        hx: Optional initial hidden state of shape (num_layers, batch, hidden_size).
            If not provided, defaults to zeros.

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for the input sequence.

      where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}
    """
    assert not isinstance(input, PackedSequence), \
      "PackedSequence is not supported. Use a regular tensor instead."

    if input.dim() not in (2, 3):
      raise ValueError(
          f"GRU: Expected input to be 2D or 3D, got {input.dim()}D instead")
    is_batched = input.dim() == 3
    batch_dim = 0 if self.batch_first else 1

    # Unsqueeze the input to (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size) if unbatched.
    if not is_batched:
      input = input.unsqueeze(batch_dim)
      if hx is not None:
        if hx.dim() != 2:
          raise RuntimeError(
              f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
          )
        hx = hx.unsqueeze(1)
    else:
      if hx is not None and hx.dim() != 3:
        raise RuntimeError(
            f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
        )

    batch_size = input.size(0) if self.batch_first else input.size(1)
    if hx is None:
      hx = input.new_zeros(
          self.num_layers,
          batch_size,
          self.hidden_size,
          dtype=input.dtype,
          device=input.device,
      )

    self.check_forward_args(input, hx, None)

    # Reshape the input to (seq_len, batch_size, input_size) if batch is at the first dimension.
    if self.batch_first:
      input = input.transpose(0, 1)

    # The output of one layer is the input to the next.
    output = input
    hidden_states = []

    # Loop over each layer.
    for layer in range(self.num_layers):
      init = {
          'h': hx[layer],
          'w_ih': getattr(self, f'weight_ih_l{layer}'),
          'w_hh': getattr(self, f'weight_hh_l{layer}')
      }
      if self.bias:
        init['b_ih'] = getattr(self, f'bias_ih_l{layer}', None)
        init['b_hh'] = getattr(self, f'bias_hh_l{layer}', None)

      # Use scan to iterate over the time dimension.
      # Here, scan(fn, init, xs) applies step_fn to each time slice of `output`.
      final_carry, layer_output = scan(
          fn=_gru_step_fn, init=init, xs=output, is_fn_pure=True)
      hidden_states.append(final_carry['h'])
      # Apply dropout on the output of the current layer (if not the final layer).
      if layer < self.num_layers - 1 and self.dropout > 0:
        layer_output = F.dropout(
            layer_output, p=self.dropout, training=self.training)
      output = layer_output

    # Stack the final hidden states for each layer.
    hidden = torch.stack(hidden_states, dim=0)

    # Reshape the output according to the input format. The original shape of the output is (seq_len, batch_size, hidden_size).
    # If the input was unbatched, we need to squeeze it back to (seq_len, hidden_size).
    if not is_batched:
      output = output.squeeze(1)
      if hidden is not None:
        hidden = hidden.squeeze(1)

    # If the input has batch at the first dimension, we need to transpose the output back to (batch_size, seq_len, hidden_size).
    if self.batch_first:
      output = output.transpose(0, 1)

    return output, hidden


# Define the step function for scanning over time.
# This function has to be in global scope to take advantage of the `scan` tracing cache.
#
# x_t: (batch, current_input_size)
# h: (batch, hidden_size)
# carry: dictionary containing h and weights/biases.
def _gru_step_fn(carry, x_t):
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

  carry_new = {**carry, 'h': h_new}

  return carry_new, h_new
