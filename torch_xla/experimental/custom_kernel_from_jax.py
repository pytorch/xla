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


@dataclasses.dataclass
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
  # Check more rules from MaxText config:
  # https://github.com/AI-Hypercomputer/maxtext/blob/462087ed90a60485a145e909e047bacc28397f82/MaxText/configs/base.yml#L261.
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
  AttentionType_LOCAL_SLIDING: bool = False
  SLIDE_WINDOW_SIZE: int | None = None

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
  # input q,k,v shape: [batch, #head, seq_len, kv]
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

    # Apply local masking if local sliding attention is enabled.
    if config.AttentionType_LOCAL_SLIDING:
      if config.SLIDE_WINDOW_SIZE is None:
        raise ValueError(
            "Sliding_window_size must be set if Local Sliding attention type")
      mask &= splash_attention_mask.LocalMask(
          shape=(query.shape[2], query.shape[2]),
          window_size=(config.SLIDE_WINDOW_SIZE, config.SLIDE_WINDOW_SIZE),
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


@functools.lru_cache(maxsize=16)
def _get_jax_forward_function(config_json: str, attn_logits_soft_cap,
                              has_segment_ids):
  """Cached factory function to create JAX forward functions"""
  config = SplashAttentionConfig.from_json(config_json)
  if has_segment_ids:
    return functools.partial(
        splash_attention_jax_wrapper,
        config=config,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
  else:
    return functools.partial(
        splash_attention_jax_wrapper,
        decoder_segment_ids=None,
        config=config,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )


@functools.lru_cache(maxsize=16)
def _get_jax_backward_function(config_json: str, attn_logits_soft_cap,
                               has_segment_ids):
  """Cached factory function to create JAX backward functions"""
  jax_f = _get_jax_forward_function(config_json, attn_logits_soft_cap,
                                    has_segment_ids)
  import jax

  if has_segment_ids:

    def jax_grad_f_wrapper(query, key, value, decoder_segment_ids, grad_output):
      primals, f_vjp = jax.vjp(jax_f, query, key, value, decoder_segment_ids)
      return f_vjp(grad_output)

    return jax_grad_f_wrapper
  else:

    def jax_grad_f_wrapper(query, key, value, grad_output):
      primals, f_vjp = jax.vjp(jax_f, query, key, value)
      return f_vjp(grad_output)

    return jax_grad_f_wrapper


@xp.trace_me("tpu_splash_attention_jax_call_wrapper")
def tpu_splash_attention_jax_call_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None,
    attn_logits_soft_cap: float | None = None,
    is_forward: bool = True,
    grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
  # return tuple to fit for the output num for both fwd and bwd
  query = query.contiguous()
  key = key.contiguous()
  value = value.contiguous()

  # TODO: xb.call_jax() doesn't accept the input tensor with shape size 0. We
  # have to split the decoder_segment_ids to be None or torch.Tensor cases.
  # Later we can unify those two cases once 0 size shape tensor is supported.
  has_decoder_ids = decoder_segment_ids is not None and decoder_segment_ids.shape
  input_args = ([query, key, value, decoder_segment_ids]
                if has_decoder_ids else [query, key, value])
  if is_forward:
    jax_f = _get_jax_forward_function(config, attn_logits_soft_cap,
                                      has_decoder_ids)
    output = call_jax(jax_f, input_args, {}, "splash_attention_jax_wrapper_fw")
    return (output, None, None)
  else:
    # TODO: find out a way to skip grad computation for decoder_segment_ids
    jax_grad_f = _get_jax_backward_function(config, attn_logits_soft_cap,
                                            has_decoder_ids)
    q_grad, k_grad, v_grad, *_rest = call_jax(
        jax_grad_f,
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
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return tpu_splash_attention_jax_call_wrapper(
      q,
      k,
      v,
      config,
      decoder_segment_ids,
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
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # q.shape: batch_size, seq_length, num_heads, kv (head_dim?)
  return (torch.empty_like(q), None, None)


@custom_op("xla::sa_custom_backward", mutates_args=())
def sa_custom_backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None,
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  o = tpu_splash_attention_jax_call_wrapper(
      q,
      k,
      v,
      config,
      decoder_segment_ids,
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
    attn_logits_soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))


class SplashAttention(torch.autograd.Function):

  @staticmethod
  @requires_jax
  def forward(ctx, q, k, v, config, decoder_segment_ids, attn_logits_soft_cap):
    output = sa_custom_forward(q, k, v, config, decoder_segment_ids,
                               attn_logits_soft_cap)[0]
    ctx.save_for_backward(q, k, v, decoder_segment_ids, attn_logits_soft_cap)
    ctx.config = config
    return output

  @staticmethod
  @requires_jax
  def backward(ctx, grad_output):
    q, k, v, decoder_segment_ids, attn_logits_soft_cap = ctx.saved_tensors
    config = ctx.config
    grad_q, grad_k, grad_v = sa_custom_backward(grad_output, q, k, v, config,
                                                decoder_segment_ids,
                                                attn_logits_soft_cap)
    return grad_q, grad_k, grad_v, None, None, None


def splash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None = None,
    attn_logits_soft_cap: float | None = None,
) -> torch.Tensor:
  return SplashAttention.apply(q, k, v, config, decoder_segment_ids,
                               attn_logits_soft_cap)
