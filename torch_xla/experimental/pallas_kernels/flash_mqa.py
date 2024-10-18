"""Example flash MQA TPU kernel."""
import functools

import jax
from jax import lax
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def when(condition):
  return lambda f: jax.lax.cond(condition, f, lambda: None)


def flash_attention_kernel(*args, **kwargs):
  nb, nh = args[0].shape[:2]
  for ib in range(nb):
    for ih in range(nh):
      flash_attention_kernel_unbatched((ib, ih), *args, **kwargs)


def flash_attention_kernel_unbatched(
    batch_idx,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal: bool,
    sm_scale: float,
    block_k: int,
    kv_seq_len: int,
):
  _, _, block_q, head_dim = q_tile_ref.shape
  _, block_k_major, _ = k_tile_ref.shape
  local_batch_index, _ = batch_idx

  q_seq_idx = pl.program_id(2)
  kv_seq_idx = pl.program_id(3)

  kv_major_index = kv_seq_idx * block_k_major
  q_index = q_seq_idx * block_q

  on_diag, below_or_on_diag = False, False
  if block_q == block_k_major:
    on_diag = q_seq_idx == kv_seq_idx
    below_or_on_diag = q_seq_idx >= kv_seq_idx
  else:
    q_end = (q_seq_idx + 1) * block_q
    kv_index = kv_seq_idx * block_k_major
    below_or_on_diag = q_end > kv_index
    diag_index = jax.lax.div(q_seq_idx * block_q, block_k_major)
    on_diag = kv_seq_idx == diag_index

  @when(kv_seq_idx == 0)
  def start_new_sequence():
    m_scratch_ref[:] = jnp.full(
        m_scratch_ref.shape, -jnp.inf, dtype=jnp.float32
    )
    l_scratch_ref[:] = jnp.zeros(l_scratch_ref.shape, dtype=jnp.float32)
    acc_scratch_ref[:, :] = jnp.zeros(o_tile_ref.shape[2:], dtype=jnp.float32)

  def body(i, refs):
    kv_index = kv_major_index + i * block_k

    def run_iter():
      () = refs
      m_i = m_scratch_ref[:]
      l_i = l_scratch_ref[:]
      start_k = pl.multiple_of(i * block_k, block_k)
      q = q_tile_ref[batch_idx].astype(jnp.float32)
      k = pl.load(
          k_tile_ref, (local_batch_index, pl.dslice(start_k, block_k),
                       pl.dslice(None))
      ).astype(jnp.float32)

      p_ij = pl.dot(q, k, trans_b=True)  # [block_q, block_k]
      if sm_scale != 1.0:
        p_ij *= sm_scale

      if causal:
        q_span = q_index + jax.lax.broadcasted_iota(
            jnp.int32, (block_q, block_k), 0
        )
        kv_span = kv_index + jax.lax.broadcasted_iota(
            jnp.int32, (block_q, block_k), 1
        )
        causal_mask = jnp.where(q_span < kv_span, float("-inf"), 0.)
        p_ij = p_ij + causal_mask

      m_ij = jnp.max(p_ij, axis=1)[:, None]  # Row max, shape [block_q, 1].
      p_ij = jnp.exp(p_ij - m_ij)  # Shape [block_q, block_k].

      if causal and block_q > block_k:
        # If we have skinny blocks, we might have rows that are entirely
        # -inf. We need to mask out the nans that are created as a result
        # TODO(sharadmv,apaszke): enable this nan mask here
        # p_ij = jnp.where(jnp.isnan(p_ij), 0., p_ij)
        raise NotImplementedError

      m_i_new = jnp.maximum(m_i, m_ij)  # Shape [block_q, 128].
      alpha = jnp.exp(m_i - m_i_new)  # Shape [block_q, 128].
      beta = jnp.exp(m_ij - m_i_new)  # Shape [block_q, 128].

      l_ij = jnp.sum(p_ij, axis=1)[:, None]  # Shape [block_q, 1].
      l_i_new = alpha * l_i + beta * l_ij  # Shape [block_q, 128].
      p_scale = beta / l_i_new  # Shape [block_q, 128].
      p_scale_repeats, rem = divmod(block_k, 128)
      if rem != 0:
        raise NotImplementedError("block_k should be a multiple of 128")
      p_ij = p_ij * pltpu.repeat(p_scale, p_scale_repeats, axis=1)
      acc_scale = l_i / l_i_new * alpha  # Shape [block_q, 128].

      acc_scale_repeats, rem = divmod(head_dim, 128)
      if rem != 0:
        raise NotImplementedError("head_dim should be a multiple of 128")
      acc_scratch_ref[:] *= pltpu.repeat(acc_scale, acc_scale_repeats, axis=1)

      # Update m_i and l_i for the next block_k.
      l_scratch_ref[:] = l_i_new
      m_scratch_ref[:] = m_i_new

      # Add the new block of attention weights.
      v = pl.load(
          v_tile_ref, (local_batch_index, pl.dslice(start_k, block_k),
                       pl.dslice(None))
      ).astype(jnp.float32)
      acc_scratch_ref[:] += jnp.dot(p_ij, v)

    if causal:
      should_run_iter = (q_seq_idx + 1) * block_q > kv_index
      when(should_run_iter)(run_iter)
    else:
      run_iter()

  if causal:
    @when(below_or_on_diag)
    def _run_body():
      for_loop.for_loop(block_k_major // block_k, body, init_state=())
  else:
    for_loop.for_loop(block_k_major // block_k, body, init_state=())

  if causal:
    @when(on_diag)
    def store_output():
      o_tile_ref[batch_idx] = acc_scratch_ref[:].astype(o_tile_ref.dtype)
  else:
    @when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
      o_tile_ref[batch_idx] = acc_scratch_ref[:].astype(o_tile_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal", "sm_scale", "block_b", "block_q", "block_k_major", "block_k",
        "debug", "interpret"
    ],
)
def flash_mqa(
    q,  # [batch_size, num_heads, seq_len, d_model]
    k,  # [batch_size, seq_len, d_model]
    v,  # [batch_size, seq_len, d_model]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_b: int = 1,
    block_q: int = 128,
    block_k_major: int = 128,
    block_k: int = 128,
    debug: bool = False,
    interpret: bool = False,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, kv_seq_len, _ = k.shape

  if block_b > batch_size:
    raise ValueError(f"{block_b=} should be smaller or equal to {batch_size=}")
  if block_q > q_seq_len:
    raise ValueError(f"{block_q=} should be smaller or equal to {q_seq_len=}")
  if block_k > kv_seq_len:
    raise ValueError(f"{block_k=} should be smaller or equal to {kv_seq_len=}")
  if block_k_major > kv_seq_len:
    raise ValueError(
        f"{block_k_major=} should be smaller or equal to {kv_seq_len=}"
    )
  if block_k_major < block_k:
    raise ValueError(f"{block_k_major=} should be smaller than {block_k=}")
  grid = (
      batch_size // block_b,
      num_heads,
      q_seq_len // block_q,
      kv_seq_len // block_k_major,
  )

  def kv_index_map(batch_index, _, q_seq_index, kv_seq_index):
    if not causal:
      return (batch_index, kv_seq_index, 0)
    q_end = (q_seq_index + 1) * block_q
    kv_index = kv_seq_index * block_k_major
    if block_q == block_k_major:
      default_index = q_seq_index
    else:
      default_index = jax.lax.div(q_seq_index * block_q, block_k_major)
    def _below_or_on_diag():
      return (batch_index, kv_seq_index, 0)
    def _above_diag():
      return (batch_index, default_index, 0)
    return lax.cond(q_end > kv_index, _below_or_on_diag, _above_diag)

  def qo_index_map(batch_index, head_index, q_seq_idx, _):
    return (batch_index, head_index, q_seq_idx, 0)

  kernel = functools.partial(
      flash_attention_kernel,
      causal=causal,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  m_scratch = jax.ShapeDtypeStruct((block_q, 128), dtype=jnp.float32)
  l_scratch = jax.ShapeDtypeStruct((block_q, 128), dtype=jnp.float32)
  acc_scratch = jax.ShapeDtypeStruct((block_q, head_dim), dtype=jnp.float32)
  with jax.named_scope(f"flash_mqa_{causal=}_{block_q=}"
                       f"_{block_k_major=}_{block_k=}"):
    return pl.pallas_call(
        kernel,
        out_shape=(out_shape, m_scratch, l_scratch, acc_scratch),
        in_specs=[
            pl.BlockSpec((block_b, 1, block_q, head_dim), qo_index_map),
            pl.BlockSpec((block_b, block_k_major, head_dim), kv_index_map),
            pl.BlockSpec((block_b, block_k_major, head_dim), kv_index_map),
        ],
        out_specs=[
            pl.BlockSpec((block_b, 1, block_q, head_dim), qo_index_map),
            pl.BlockSpec(m_scratch.shape, lambda *_: (0, 0)),
            pl.BlockSpec(l_scratch.shape, lambda *_: (0, 0)),
            pl.BlockSpec(acc_scratch.shape, lambda *_: (0, 0)),
        ],
        grid=grid,
        debug=debug,
        interpret=interpret,
    )(q, k, v)[0]


@functools.partial(jax.jit, static_argnames=["sm_scale", "causal"])
@jax.default_matmul_precision("bfloat16")
def mqa_reference(q, k, v, sm_scale: float = 1.0, causal: bool = False):
  logits = jnp.einsum(
      "bhqc,bkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
      _dot_general=functools.partial(
          lax.dot_general, preferred_element_type=jnp.float32,
      ),
      precision=jax.lax.Precision.DEFAULT,
  ).astype(jnp.float32)
  if causal:
    mask = jnp.tril(jnp.ones((1, 1, q.shape[2], k.shape[1]), dtype=bool))
    mask = jnp.broadcast_to(mask, logits.shape)
    logits = jnp.where(mask, logits, float("-inf"))
  weights = jax.nn.softmax(logits * sm_scale, axis=-1)
  return jnp.einsum("bhqk,bkc->bhqc", weights, v.astype(jnp.float32)).astype(
      q.dtype
  )
