import sys
import os

example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import math


def apply_xla_flash_attention(query_states, key_states, value_states):
  from torch_xla.experimental.custom_kernel import flash_attention

  # q, k, v should all have the shape [B, n_head, S, head_dim]
  head_dim = query_states.size()[-1]
  query_states = query_states / math.sqrt(head_dim)
  # Our simplified version of decoder only model does not use any mask.
  attn_output = flash_attention(
      query_states, key_states, value_states, causal=False)
  return attn_output


class TrainDecoderOnlyFlashAttention(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    self.config.use_flash_attention = True
    for layer in self.model.layers:
      layer.self_attn.flash_attention_impl = apply_xla_flash_attention


if __name__ == '__main__':
  fa = TrainDecoderOnlyFlashAttention()
  fa.start_training()
