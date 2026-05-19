from typing_extensions import override
from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel


class DecoderWithScan(DecoderOnlyModel):

  def __init__(self, config: DecoderOnlyConfig):
    super().__init__(config)
    self.config = config

  @override
  def run_decoder_layers(self, hidden_states):
    from torch_xla.experimental.scan_layers import scan_layers
    return scan_layers(
        self.layers,
        hidden_states,
        is_layer_pure=self.config.is_decoder_layer_pure)
