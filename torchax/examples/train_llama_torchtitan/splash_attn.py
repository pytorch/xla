import functools

import jax
import jax.numpy as jnp

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map


def tpu_splash_attention(
    mesh,
    q_sharding,
    # Input should be of shape (batch, length, heads, kv_dim)
    apply_shard_map,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    decoder_segment_ids: jax.Array | None,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  """TPU Flash Attention."""
  if decoder_segment_ids is not None:
    decoder_segment_ids = splash_attention_kernel.SegmentIds(
        decoder_segment_ids, decoder_segment_ids)

  print('HERE', locals())

  global_block_q = 1024
  global_block_kv = 512
  global_block_kv_compute = 512
  global_block_q_dkv = 2048
  global_block_kv_dkv = 512
  global_block_kv_dkv_compute = 512
  global_block_q_dq = 2048
  global_block_kv_dq = 512
  global_use_fused_bwd_kernel = False
  global_q_layout = 'HEAD_DIM_MINOR'
  global_k_layout = 'HEAD_DIM_MINOR'
  global_v_layout = 'HEAD_DIM_MINOR'

  def wrap_flash_attention(query, key, value, decoder_segment_ids):
    if decoder_segment_ids is not None:
      assert (
          query.shape[2] == decoder_segment_ids.q.shape[1]
      ), "Sharding along sequence dimension not allowed in tpu kernel attention"
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(global_block_q, query.shape[2]),
        block_kv=min(global_block_kv, key.shape[2]),
        block_kv_compute=min(global_block_kv_compute, key.shape[2]),
        block_q_dkv=min(global_block_q_dkv, query.shape[2]),
        block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
        block_q_dq=None if global_use_fused_bwd_kernel else min(
            global_block_q_dq, query.shape[2]),
        block_kv_dq=None if global_use_fused_bwd_kernel else min(
            global_block_kv_dq, query.shape[2]),
        use_fused_bwd_kernel=global_use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
        k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
        v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
    )

    mask = splash_attention_mask.CausalMask(
        shape=(query.shape[2], query.shape[2]))

    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(mask,) * query.shape[1])
    #splash_kernel = splash_attention_kernel.make_splash_mha(
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )

    return jax.vmap(splash_kernel)(
        query, key, value, segment_ids=decoder_segment_ids)

  if apply_shard_map:
    wrap_flash_attention = shard_map(
        wrap_flash_attention,
        mesh=mesh,
        in_specs=(
            q_sharding,
            q_sharding,
            q_sharding,
            None,
        ),
        out_specs=q_sharding,
        check_rep=False,
    )

  x = wrap_flash_attention(query, key, value, decoder_segment_ids)
  return x


if __name__ == '__main__':
  main()
