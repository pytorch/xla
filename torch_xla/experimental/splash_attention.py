import dataclasses
import functools
import json
from dataclasses import asdict
from typing import Any

import torch
import torch_xla.debug.profiler as xp
from torch.library import custom_op
from torch.utils._pytree import tree_flatten
from torch_xla.core.xla_builder import call_jax
from torch_xla.distributed.spmd import Mesh
from torch_xla.experimental.custom_kernel import requires_jax


@dataclasses.dataclass(eq=True, frozen=True)
class SplashAttentionConfig:
  ### Splash attention block sizes
  # These can be tuned for specific hardware generations, and can be set up to
  # the model's sequence length.
  sa_block_q: int = 2048
  sa_block_kv: int = 2048
  sa_block_kv_compute: int = 2048
  sa_block_q_dkv: int = 2048
  sa_block_kv_dkv: int = 2048
  sa_block_kv_dkv_compute: int = 2048
  sa_block_q_dq: int = 2048
  sa_block_kv_dq: int = 2048
  sa_use_fused_bwd_kernel: bool = True
  sa_q_layout: str = "HEAD_DIM_MINOR"
  sa_k_layout: str = "HEAD_DIM_MINOR"
  sa_v_layout: str = "HEAD_DIM_MINOR"
  mesh: str | None = None
  qkv_partition_spec: tuple[tuple[str] | str | None] = (
      ("data", "fsdp"),
      None,
      None,
      None,
  )
  segment_ids_partition_spec: tuple[tuple[str] | str | None] = (
      ("data", "fsdp"),
      None,
  )
  attentiontype_local_sliding: bool = False
  slide_window_size: int | None = None

  def to_json(self) -> str:
    """Serialize to JSON string"""
    return json.dumps(asdict(self))

  @classmethod
  def from_json(cls, data: str) -> "SplashAttentionConfig":
    """Deserialize from JSON string"""
    json_data = json.loads(data)

    # Define a function to convert lists to tuples
    def list_to_tuple(x):
      if isinstance(x, list):
        return tuple(list_to_tuple(item) for item in x)
      return x

    # Apply the conversion to all fields
    converted_data = {k: list_to_tuple(v) for k, v in json_data.items()}
    return SplashAttentionConfig(**converted_data)

  @requires_jax
  def maybe_convert_and_get_jax_mesh(self):
    # Construct a JAX mesh object with the same device ids shape and ordering
    # from torch_xla device mesh.
    mesh = Mesh.from_str(self.mesh)
    import jax
    import numpy as np
    from jax._src import mesh as mesh_lib

    assert mesh.axis_names is not None, "Omitting axis names is not yet supported"

    # Create a mapping from device ID to device object
    all_devices = jax.devices()
    device_id_to_device = {device.id: device for device in all_devices}
    device_ids_array = mesh.device_ids.reshape(*mesh.mesh_shape)
    device_array = np.empty(device_ids_array.shape, dtype=object)
    for idx in np.ndindex(device_ids_array.shape):
      device_id = device_ids_array[idx]
      if device_id in device_id_to_device:
        device_array[idx] = device_id_to_device[device_id]
      else:
        raise ValueError(
            f"torch_xla device ID {device_id} not found in available JAX devices"
        )
    return mesh_lib.Mesh(device_array, axis_names=mesh.axis_names)


@xp.trace_me("splash_attention_kernel_wrapper")
def splash_attention_jax_wrapper(
    query,
    key,
    value,
    decoder_segment_ids,
    causal: bool,
    config: SplashAttentionConfig,
    attn_logits_soft_cap,
):
  """Splash attention kernel wrapper for JAX
  Inside the function, everything is JAX specific. We convert the torch_xla mesh
  and partition spec into jax specific format, and reuse the MaxText attention
  call function from
  https://github.com/AI-Hypercomputer/maxtext/blob/d8ffb5c4fc65e6832976226a8053236c2fe3164e/MaxText/layers/attentions.py#L336-L430.
  """
  import jax
  from jax.experimental import shard_map
  from jax.experimental.pallas.ops.tpu.splash_attention import (
      splash_attention_kernel,
      splash_attention_mask,
  )
  mesh = config.maybe_convert_and_get_jax_mesh()
  # input q,k,v shape: [batch, #head, seq_len, head_dim]
  if decoder_segment_ids is not None and not decoder_segment_ids.shape:
    decoder_segment_ids = None
  if decoder_segment_ids is not None:
    decoder_segment_ids = splash_attention_kernel.SegmentIds(
        decoder_segment_ids, decoder_segment_ids)
  axis_names = jax.sharding.PartitionSpec(*config.qkv_partition_spec)
  segment_axis_names = jax.sharding.PartitionSpec(
      *config.segment_ids_partition_spec)

  global_block_q = config.sa_block_q
  global_block_kv = config.sa_block_kv
  global_block_kv_compute = config.sa_block_kv_compute
  global_block_q_dkv = config.sa_block_q_dkv
  global_block_kv_dkv = config.sa_block_kv_dkv
  global_block_kv_dkv_compute = config.sa_block_kv_dkv_compute
  global_block_q_dq = config.sa_block_q_dq
  global_block_kv_dq = config.sa_block_kv_dq
  global_use_fused_bwd_kernel = config.sa_use_fused_bwd_kernel
  global_q_layout = config.sa_q_layout
  global_k_layout = config.sa_k_layout
  global_v_layout = config.sa_v_layout
  shard_map = shard_map.shard_map

  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(
          axis_names,
          axis_names,
          axis_names,
          segment_axis_names,
      ),
      out_specs=axis_names,
      check_rep=False,
  )
  def wrap_flash_attention(query, key, value, decoder_segment_ids):
    seq_len = query.shape[2]
    if decoder_segment_ids is not None:
      assert (
          seq_len == decoder_segment_ids.q.shape[1]
      ), "Sharding along sequence dimension not allowed in tpu kernel attention"
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(global_block_q, seq_len),
        block_kv=min(global_block_kv, key.shape[2]),
        block_kv_compute=min(global_block_kv_compute, key.shape[2]),
        block_q_dkv=min(global_block_q_dkv, seq_len),
        block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(global_block_kv_dkv_compute, seq_len),
        block_q_dq=None if global_use_fused_bwd_kernel else min(
            global_block_q_dq, seq_len),
        block_kv_dq=None if global_use_fused_bwd_kernel else min(
            global_block_kv_dq, seq_len),
        use_fused_bwd_kernel=global_use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
        k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
        v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
    )
    if not causal:
      mask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
    else:
      mask = splash_attention_mask.FullMask(_shape=(seq_len, seq_len))

    # Apply local masking if local sliding attention is enabled.
    if config.attentiontype_local_sliding:
      if config.slide_window_size is None:
        raise ValueError(
            "Sliding_window_size must be set if Local Sliding attention type")
      mask &= splash_attention_mask.LocalMask(
          shape=(seq_len, seq_len),
          window_size=(config.slide_window_size, config.slide_window_size),
          offset=0,
      )

    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(mask,) * query.shape[1])
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    return jax.vmap(splash_kernel)(
        query, key, value, segment_ids=decoder_segment_ids)

  devices_in_data_fsdp = mesh.shape["data"] * mesh.shape["fsdp"]
  assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
      "Batch dimension should be shardable among the devices in data and fsdp axis"
  )
  x = wrap_flash_attention(query, key, value, decoder_segment_ids)
  # x.shape = [batch, heads, seq_length, head_dim]
  return x


@requires_jax
def _jax_grad_f(query, key, value, decoder_segment_ids, causal, config,
                attn_logits_soft_cap, grad_output):
  import jax
  differentiated_fun = functools.partial(
      splash_attention_jax_wrapper,
      decoder_segment_ids=decoder_segment_ids,
      causal=causal,
      config=config,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )
  primals, f_vjp = jax.vjp(differentiated_fun, query, key, value)
  return f_vjp(grad_output)


@xp.trace_me("tpu_splash_attention_jax_call_wrapper")
def tpu_splash_attention_jax_call_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: SplashAttentionConfig,
    decoder_segment_ids: torch.Tensor | None,
    causal: bool,
    attn_logits_soft_cap: float | None = None,
    is_forward: bool = True,
    grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
  # return tuple to fit for the output num for both fwd and bwd
  query = query.contiguous()
  key = key.contiguous()
  value = value.contiguous()
  input_args = [
      query, key, value, decoder_segment_ids, causal, config,
      attn_logits_soft_cap
  ]
  if is_forward:
    output = call_jax(splash_attention_jax_wrapper, input_args, {},
                      "splash_attention_jax_wrapper_fw")
    return (output, None, None)
  else:
    # TODO: find out a way to skip grad computation for decoder_segment_ids
    q_grad, k_grad, v_grad, *_rest = call_jax(
        _jax_grad_f,
        input_args + [grad_output],
        {},
        "splash_attention_jax_wrapper_bw",
    )
    return (q_grad, k_grad, v_grad)


@custom_op("xla::sa_custom_forward", mutates_args=())
def sa_custom_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None,
    causal: bool | None,
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  config = SplashAttentionConfig.from_json(config)
  return tpu_splash_attention_jax_call_wrapper(
      q,
      k,
      v,
      config,
      decoder_segment_ids,
      causal,
      attn_logits_soft_cap,
      is_forward=True,
      grad_output=None,
  )


@sa_custom_forward.register_fake
def sa_custom_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None,
    causal: bool | None,
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # q.shape: batch_size, seq_length, num_heads, head_dim
  return (torch.empty_like(q), None, None)


@custom_op("xla::sa_custom_backward", mutates_args=())
def sa_custom_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None,
    causal: bool | None,
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  config = SplashAttentionConfig.from_json(config)
  o = tpu_splash_attention_jax_call_wrapper(
      q,
      k,
      v,
      config,
      decoder_segment_ids,
      causal,
      attn_logits_soft_cap,
      is_forward=False,
      grad_output=grad_output,
  )
  return o


@sa_custom_backward.register_fake
def sa_custom_backward_fake(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None,
    causal: bool | None,
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))


class SplashAttention(torch.autograd.Function):

  @staticmethod
  @requires_jax
  def forward(ctx, q, k, v, config, decoder_segment_ids, causal,
              attn_logits_soft_cap):
    output = sa_custom_forward(q, k, v, config, decoder_segment_ids, causal,
                               attn_logits_soft_cap)[0]
    ctx.save_for_backward(q, k, v, decoder_segment_ids, attn_logits_soft_cap)
    ctx.config = config
    ctx.causal = causal
    return output

  @staticmethod
  @requires_jax
  def backward(ctx, grad_output):
    q, k, v, decoder_segment_ids, attn_logits_soft_cap = ctx.saved_tensors
    config = ctx.config
    causal = ctx.causal
    grad_q, grad_k, grad_v = sa_custom_backward(grad_output, q, k, v, config,
                                                decoder_segment_ids, causal,
                                                attn_logits_soft_cap)
    return grad_q, grad_k, grad_v, None, None, None, None


def splash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None = None,
    causal: bool = True,
    attn_logits_soft_cap: float | None = None,
) -> torch.Tensor:
  """Splash attention function.
  
  Args:
    decoder_segment_ids: Segment ids are a pair of 1D jax.Arrays, one for Q (of
    size q_seq_len) and one for KV (of size kv_seq_len).  A segment id mask is
    computed such that only tokens that have the same segment id can attend to
    each other. This creates a block-sparse pattern along the main diagonal.
    attn_logits_soft_cap: The soft clipping value for logits pre softmax.

  Returns:
    The attention output tensor.
  """
  return SplashAttention.apply(q, k, v, config, decoder_segment_ids, causal,
                               attn_logits_soft_cap)
