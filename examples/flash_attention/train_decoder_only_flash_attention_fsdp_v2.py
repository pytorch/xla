import sys
import os
example_fsdp_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))) + '/fsdp'
sys.path.append(example_fsdp_folder)
from train_decoder_only_fsdp_v2 import TrainDecoderOnlyFSDPv2

import math

from torch_xla import runtime as xr

def apply_xla_flash_attention_with_spmd(query_states, key_states, value_states):
    from torch_xla.experimental.custom_kernel import flash_attention

    # q, k, v should all have the shape [B, n_head, S, head_dim]
    head_dim = query_states.size()[-1]
    query_states = query_states / math.sqrt(head_dim)

    # Our simplified version of decoder only model does not use any mask.
    # flash_attention will use the global_mesh set in the TrainDecoderOnlyFSDPv2.
    attn_output = flash_attention(
        query_states, key_states, value_states, causal=False, partition_spec=('fsdp', None, None, None))
    return attn_output

class TrainDecoderOnlyFlashAttentionFSDPv2(TrainDecoderOnlyFSDPv2):
    def __init__(self):
        super().__init__()

        self.config.use_flash_attention = True
        for layer in self.model.layers:
            layer.self_attn.flash_attention_impl = apply_xla_flash_attention_with_spmd

if __name__ == '__main__':
  # Enable the SPMD
  xr.use_spmd()
  fa_fsdp = TrainDecoderOnlyFlashAttentionFSDPv2()
  fa_fsdp.start_training()
