# v1 of the extended paged attention kernel.
# do repeat_interleave
# for b_idx in range(batch_size):
#   for query_head_idx in range(num_query_head):
#     # q.shape=[query_len, head_size]
#     # k.shape=[kv_len, head_size]
#     # attn=[query_len, kv_len]
#     # v.shape=[kv_len, head_size]\
#     # out.shape=[query_len, head_size

"""PagedAttention TPU kernel."""

from collections.abc import Sequence
import functools
from typing import Literal, cast

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref, # [num_kv_heads, total_num_pages, page_size, head_dim]
      scales_pages_hbm_ref,
      vmem_buffer, # [pages_per_compute_block, page_size, head_dim]
      scales_vmem_buffer,
      sem,
      page_indices,
      page_indices_start_offset,
      num_pages_to_load,
      kv_head_index,
  ):
    # Original k_pages has shape [num_kv_heads, total_num_pages, page_size, head_dim]
    self._vmem_buffer = vmem_buffer
    self._scales_vmem_buffer = scales_vmem_buffer
    self._num_pages_to_load = num_pages_to_load
    if kv_head_index is not None:
      self._pages_hbm_ref = pages_hbm_ref.at[kv_head_index]
      if scales_pages_hbm_ref is not None:
        self._scales_pages_hbm_ref = scales_pages_hbm_ref.at[kv_head_index]
      else:
        self._scales_pages_hbm_ref = None
    else:
      self._pages_hbm_ref = pages_hbm_ref
      self._scales_pages_hbm_ref = scales_pages_hbm_ref
    self._sem = sem
    self._page_indices = page_indices
    self._page_indices_start_offset = page_indices_start_offset
    self._async_copies = [
        self._make_async_copy(i) for i in range(self._num_pages_to_load)
    ]
    if (
        self._scales_pages_hbm_ref is not None
        and self._scales_vmem_buffer is not None
    ):
      self._async_copies += [
          self._make_scales_async_copy(i)
          for i in range(self._num_pages_to_load)
      ]

  def _make_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    # pl.debug_print('xw32 line74 _make_async_copy. page_index={}', page_index)
    return pltpu.make_async_copy(
        self._pages_hbm_ref.at[page_index], self._vmem_buffer.at[i], self._sem
    )

  def _make_scales_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    return pltpu.make_async_copy(
        self._scales_pages_hbm_ref.at[page_index],  # pytype: disable=attribute-error
        self._scales_vmem_buffer.at[i],  # pytype: disable=attribute-error
        self._sem,
    )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def _maybe_dequantize(self, x, x_scale, dtype=jnp.bfloat16):
    if x_scale is None:
      return x.astype(dtype)
    return quantization_utils.from_int8(x, x_scale, dtype=dtype)

  def wait_and_get_loaded(self) -> jax.Array:
    """Wait async copies and gets the loaded buffer as a jax.Array."""
    # Return value shape is (pages_per_compute_block*page_size,head_dim)
    for async_copy in self._async_copies:
      async_copy.wait()
    head_dim = self._vmem_buffer.shape[-1]
    jax_array = self._vmem_buffer[...].astype(jnp.float32)
    if self._scales_vmem_buffer is not None:
      scales_jax_array = self._scales_vmem_buffer[...].astype(jnp.float32)
    else:
      scales_jax_array = None
    jax_array = self._maybe_dequantize(jax_array, scales_jax_array)
    return jax_array.reshape(-1, head_dim)

# TODO(xw32): should add http://google3/third_party/py/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py;l=224;rcl=671066741
def _flash_attention(
    q_head_idx,  # scalar, ranges from 0 to num_query_heads_per_kv_head
    lengths_ref,  # [batch_size] jax.Array the length of each example
    page_indices_ref,  # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[batch_size, pages_per_sequence]
    # input
    q_ref,  # q_ref.shape=[1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    k,  # Should be [pages_per_compute_block*page_size,head_dim]
    v,  # Should be [pages_per_compute_block*page_size,head_dim]
    # output
    o_ref,  # same shape as q_ref: [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    l_ref,  # (1, num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE)
    m_ref,  # (1, num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE)
    o_debug_ref,  # [batch_size, num_q_heads, query_len, head_dim]
    l_scratch_ref,  # [num_queries_per_compute_block, MIN_BLOCK_SIZE]
    m_scratch_ref,  # [num_queries_per_compute_block, MIN_BLOCK_SIZE]
    acc_scratch_ref,  # [num_queries_per_compute_block, head_dim]
    *,
    batch_size: int,
    num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    query_len: int,
    page_size: int,
    head_dim: int,
):
  b, kv_head_idx, q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
      pl.program_id(3),
  )
  pl.debug_print('xw32 line135 _flash_attention begins, b={}, kv_head_idx={}, q_blk_idx={}, kv_blk_idx={}',
                 b, kv_head_idx, q_blk_idx, kv_blk_idx)
  print(f'xw32 line145 {l_scratch_ref.shape=}, {m_scratch_ref.shape=}, {acc_scratch_ref.shape=}')

  @pl.when(kv_blk_idx == 0)
  def start_new_sequence():
    pl.debug_print('xw32 line148 start_new_sequence')
    l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, jnp.float32)
    m_scratch_ref[...] = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
    acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  m_prev = m_scratch_ref[...]
  l_prev = l_scratch_ref[...]
  q = q_ref[0, q_head_idx, :, :].astype(jnp.float32)  # [block_q, head_dim]
  assert q.shape == (num_queries_per_compute_block, head_dim)
  kv_seq_len_per_kv_compute_blk = num_kv_pages_per_compute_block*page_size
  assert k.shape == (kv_seq_len_per_kv_compute_blk, head_dim)
  s = jnp.einsum('hd,td->ht', q, k, preferred_element_type=jnp.float32)  # [block_q, block_k]
  assert s.shape == (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk)

  q_index = q_blk_idx * num_queries_per_compute_block
  kv_index = kv_blk_idx * kv_seq_len_per_kv_compute_blk
  kv_len = lengths_ref[b]
  pl.debug_print('xw32 line164 kv_len={}, query_len={}', kv_len, query_len)
  # old mask begins
  row_ids = (kv_len - query_len) + q_index + jax.lax.broadcasted_iota(
      jnp.int32, (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk), 0
  )
  col_ids = kv_index + jax.lax.broadcasted_iota(
      jnp.int32, (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk), 1
  )
  causal_mask = jnp.where(row_ids < col_ids, mask_value, 0.)
  # old mask ends
  # mask v1 begins
  # mask = 
  # mask v1 ends
  assert causal_mask.shape == (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk)
  s = s + causal_mask  # [block_q, block_k]
  assert s.shape == (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk)

  m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
  m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

  block_k_repeats, rem = divmod(kv_seq_len_per_kv_compute_blk, MIN_BLOCK_SIZE)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len_per_kv_compute_blk=} should be a multiple of {MIN_BLOCK_SIZE}"
    )
  p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))  # Shape [block_q, block_k]

  alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128]

  l_corr =alpha * l_prev

  l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

  # TODO(xiowei): need to have a test that uses head_dim==256
  head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
  l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
  if rem:
    if head_dim_repeats == 0:
      l_broadcast = lambda l: l[:, :head_dim]
    else:
      raise NotImplementedError(
          f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
      )
  # Need to store these l_next and m_next which will relay to the output.
  l_scratch_ref[...] = l_next
  m_scratch_ref[...] = m_next

  l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
  acc_scratch_ref[...] *= l_broadcast(l_corr * l_next_inv_safe)
  print(f'xw32 line204 {p.shape=}, {v.shape=}')
  # TODO: Here p.shape[0](block_q) has to be multiple 16. Check if we add a new
  # constraint at the beginning of the kernel.
  o_curr = jax.lax.dot(
      p.astype(v.dtype), v, preferred_element_type=jnp.float32
  )
  acc_scratch_ref[...] += o_curr * l_broadcast(l_next_inv_safe)

  num_kv_blks = pl.num_programs(3)
  print(f'xw32 line216 {o_ref.shape=}, {o_ref[0, q_head_idx].shape=}, {acc_scratch_ref[...].shape=}')  # (1, 1, 16, 128)
  pl.debug_print('xw32 line215 kv_blk_idx={}, (kv_len // kv_seq_len_per_kv_compute_blk) - 1)={}, pl.num_programs(3)={}', kv_blk_idx, (kv_len // kv_seq_len_per_kv_compute_blk) - 1, pl.num_programs(3))
  # Note num_kv_blks == pl.num_programs(3), but
  # (kv_len // kv_seq_len_per_kv_compute_blk) != pl.num_programs(3)
  # @pl.when(kv_blk_idx == num_kv_blks - 1)
  # def store_output():
  o_ref[0, q_head_idx] = acc_scratch_ref[...].astype(o_ref.dtype)
  l_ref[0, q_head_idx] = l_scratch_ref[...].astype(l_ref.dtype)
  m_ref[0, q_head_idx] = m_scratch_ref[...].astype(m_ref.dtype)
  # o_debug_ref[b, kv_head_idx, :, :] = acc_scratch_ref[...].astype(o_ref.dtype)

def paged_flash_attention_kernel(
    # prefetched value
    lengths_ref,  # [batch_size] jax.Array the length of each example
    # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[batch_size, pages_per_sequence]
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    # input
    # At caller, q.shape=[batch_size, num_q_heads query_len, head_dim]
    q_ref,  # q_ref.shape=[1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    k_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    v_scales_pages_hbm_ref,
    # output
    # same shape as q_ref: [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim], output
    o_ref,
    l_ref,
    m_ref,
    k_debug_ref,  # TODO: debug use. delete later.
    v_debug_ref,  # TODO: debug use. delete later.
    q_debug_ref,  # TODO: debug use. delete later.
    o_debug_ref,  # TODO: debug use. delete later.
    # scratch space
    k_vmem_buffer,  # shape=[2, num_kv_pages_per_compute_block, num_kv_heads, head_dim]
    k_scales_vmem_buffer,
    v_vmem_buffer,  # shape=[2, num_kv_pages_per_compute_block, num_kv_heads, head_dim]
    v_scales_vmem_buffer,
    sem,
    l_scratch_ref,
    m_scratch_ref,
    acc_scratch_ref,
    *,
    pages_per_sequence: int,  # [bs, pages_per_sequence] = page_indices.shape
    batch_size: int,
    num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int,  # TODO(xw32): consider remove it
    mask_value: float,
    attn_logits_soft_cap: float | None,
    query_len: int,
):
  """Pallas kernel for paged attention."""
  b, kv_head_idx, q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
      pl.program_id(3),
  )
  pl.debug_print('xw32 line231 b={}, kv_head_idx={}, q_blk_idx={}, kv_blk_idx={}',
                 b, kv_head_idx, q_blk_idx, kv_blk_idx)
  # pl.debug_print('xw32 pl.num_programs(0)={}, pl.num_programs(1)={}, pl.num_programs(2)={}, pl.num_programs(3)={}', pl.num_programs(0), pl.num_programs(1), pl.num_programs(2), pl.num_programs(3))
  num_q_blks = pl.num_programs(2)
  b_q_ref, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim = q_ref.shape
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages_hbm_ref.shape
  compute_blk_size_kv = page_size * num_kv_pages_per_compute_block
  kv_len = lengths_ref[b]
  # Step1: Get the K and V for the current batch and current kv head.

  @pl.when(kv_blk_idx * compute_blk_size_kv < kv_len)
  def get_kv_and_run_flash_attention():
    # Loop over num_q_heads_per_kv_head and use the same K and V
    def compute_block_indices(b, kv_head_idx, q_blk_idx, kv_blk_idx):
      """Return next_b, next_kv_head_idx, next_kv_blk_idx"""

      def advance_b():
        next_b = b + 1

        def advance_to_next_non_zero_length():
          next_next_b = next_b + 1
          return lax.fori_loop(
              next_next_b,
              batch_size,
              lambda _, b: jnp.where(lengths_ref[b] == 0, b + 1, b),
              next_next_b,
          )

        return (
            lax.cond(
                jnp.logical_and(next_b < batch_size, lengths_ref[next_b] == 0),
                advance_to_next_non_zero_length,
                lambda: next_b,
            ),
            0,  # kv_head_idx
            0,  # kv_blk_idx
        )

      def advance_kv_head_idx():
        # assumption: kv_blk_idx * compute_blk_size_kv >= lengths_ref[b]
        next_kv_head_idx = kv_head_idx + 1
        return lax.cond(q_blk_idx==num_q_blks-1,
                        lambda: lax.cond(next_kv_head_idx < num_kv_heads, lambda: (b, next_kv_head_idx, 0), advance_b),
                        lambda: (b, kv_head_idx, 0))

      return lax.cond(kv_blk_idx * compute_blk_size_kv < lengths_ref[b], lambda: (b, kv_head_idx, kv_blk_idx), advance_kv_head_idx)


    def create_kv_async_copy_descriptors(b, kv_head_idx, kv_blk_idx, buffer_index):
      # pl.debug_print('xw32 create_kv_async_copy_descriptors, kv_blk_idx={}, num_kv_pages_per_compute_block={}', kv_blk_idx, num_kv_pages_per_compute_block)
      page_offset = b * pages_per_sequence + kv_blk_idx * num_kv_pages_per_compute_block
      pages_to_load = num_kv_pages_per_compute_block
      async_copy_k = MultiPageAsyncCopyDescriptor(
          k_pages_hbm_ref,
          k_scales_pages_hbm_ref,
          k_vmem_buffer.at[buffer_index],
          k_scales_vmem_buffer.at[buffer_index]
          if k_scales_vmem_buffer is not None
          else None,
          sem,
          page_indices_ref,  # [batch_size*pages_per_sequence]
          page_offset,
          pages_to_load,
          kv_head_idx,
      )
      async_copy_v = MultiPageAsyncCopyDescriptor(
          v_pages_hbm_ref,
          v_scales_pages_hbm_ref,
          v_vmem_buffer.at[buffer_index],
          v_scales_vmem_buffer.at[buffer_index]
          if v_scales_vmem_buffer is not None
          else None,
          sem,
          page_indices_ref,
          page_offset,
          pages_to_load,
          kv_head_idx,
      )
      return async_copy_k, async_copy_v

    # copy begins
    step = step_ref[0]
    buffer_index = buffer_index_ref[0]
    @pl.when(step == 0)
    def prefetch_first_block():  # pylint: disable=unused-variable
      pl.debug_print('xw32 line318 prefetch_first_block b={}, kv_head_idx={}, kv_blk_idx={}', b, kv_head_idx, kv_blk_idx)
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          b, kv_head_idx, kv_blk_idx, buffer_index
      )
      async_copy_k.start()
      async_copy_v.start()

    next_b, next_kv_head_idx, next_kv_blk_idx = compute_block_indices(b, kv_head_idx, q_blk_idx, kv_blk_idx+1)
    pl.debug_print('xw32 line379 Calculated next_b={}, next_kv_head_idx={}, next_kv_blk_idx={}', next_b, next_kv_head_idx, next_kv_blk_idx)

    @pl.when(next_b < batch_size)
    def prefetch_next_block():  # pylint: disable=unused-variable
      pl.debug_print('xw32 line329 prefetch_next_block next_b={}, next_kv_head_idx={}, next_kv_blk_idx={}', next_b, next_kv_head_idx, next_kv_blk_idx)
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_b, next_kv_head_idx, next_kv_blk_idx, next_buffer_index
      )
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index

    pl.debug_print('xw32 line385 Retriving kv pages corresponding to b={}, kv_head_idx={}, kv_blk_idx={}', b, kv_head_idx, kv_blk_idx)
    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
      b, kv_head_idx, kv_blk_idx, buffer_index
    )
    k = async_copy_k.wait_and_get_loaded()  # (pages_per_compute_block*page_size,head_dim)
    # @pl.when(kv_head_idx == 0)
    # def _():
    #   k_f32 = k.astype(jnp.float32)
    #   val = k_f32[0, 0]
    #   val = val.astype(jnp.float32)
    #   # val = val.item()
    #   # pl.debug_print(val)
    #   # for i in range(k.shape[0]):
    #   #   for j in range(k.shape[1]):
    #   #     pl.debug_print('xw32 line345 k[{},{}]={}', i, j, k[i, j])
    v = async_copy_v.wait_and_get_loaded()
    @pl.when(kv_head_idx == 1)  # debug use. delete later.
    def _():
      k_debug_ref[...] = k.astype(jnp.float32)
      v_debug_ref[...] = v.astype(jnp.float32)
      q_debug_ref[...] = q_ref[0, 0,:,:].astype(jnp.float32)
    # copy ends
    # TODO(xw32): Temporarily, fake a k and v, remove the 2 lines below later
    # k = jnp.full((compute_blk_size_kv, head_dim), 1, dtype=jnp.float32)
    # v = jnp.full((compute_blk_size_kv, head_dim), 1, dtype=jnp.float32)

    print(f'xw32 line364 {l_ref.shape=}, {m_ref.shape=}, {l_scratch_ref.shape=}, {m_scratch_ref.shape=}, {acc_scratch_ref.shape=}')
    for q_head_idx in range(num_q_heads_per_kv_head):
      _flash_attention(
        q_head_idx,
        lengths_ref,
        page_indices_ref,
        q_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
        k,
        v,
        o_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
        l_ref,
        m_ref,
        o_debug_ref,
        l_scratch_ref,  # [num_queries_per_compute_block, MIN_BLOCK_SIZE]
        m_scratch_ref,  # [num_queries_per_compute_block, MIN_BLOCK_SIZE]
        acc_scratch_ref,  # [num_queries_per_compute_block, head_dim]
        batch_size=batch_size,
        num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
        num_queries_per_compute_block=num_queries_per_compute_block,
        pages_per_sequence=pages_per_sequence,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        query_len=query_len,
        page_size=page_size,
        head_dim=head_dim,
        )
    # o_ref.shape=[num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    step_ref[0] = step + 1
  # end of get_kv_and_run_flash_attention.

MIN_BLOCK_SIZE = 128

# @functools.partial(
#     jax.jit,
#     static_argnames=[
#         "num_kv_pages_per_compute_block",
#         "num_queries_per_compute_block",
#         "attn_logits_soft_cap",
#         "mask_value",
#         "megacore_mode",
#         "inline_seq_dim",
#     ],
# )
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array | quantization_utils.QuantizedTensor,
    v_pages: jax.Array | quantization_utils.QuantizedTensor,
    lengths: jax.Array,
    page_indices: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int = 4,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> jax.Array:
  """Paged grouped query attention.

  Args:
    q: A [batch_size, query_len, num_q_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array the length of each example.
    page_indices: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    attn_logits_soft_cap: The value used for soft capping the attention logits.
    num_kv_pages_per_compute_block: how many kv pages to be processed in one flash
      attention block in the pallas kernel.
    megacore_mode: if set, enable megacore to parallelize the computation. Must
      be one of ['kv_head', 'batch', None]. Caveat: set this only if megacore is
      enabled, otherwise the kernel may hang. If you are not sure, leave it to
      None.
      * None: disable megacore parallelism.
      * kv_head: megacore parallelism on KV heads; requires number of KV heads
        divisible by 2.
      * batch: megacore parallelism on batch dimension; requires batch divisible
        by 2.
    inline_seq_dim: whether to fuse kernel instances along the sequence dim into
      one kernel.

  Returns:
    The output of attention([batch_size, query_len, num_q_heads, head_dim]).
  """
  if isinstance(k_pages, quantization_utils.QuantizedTensor):
    k_pages, k_scales_pages = k_pages.weight, k_pages.scales
    assert isinstance(k_scales_pages, jax.Array)  # For typing.
    k_scales_pages = jnp.broadcast_to(
        k_scales_pages, (*k_scales_pages.shape[:-1], k_pages.shape[-1])
    )
  else:
    k_scales_pages = None
  if isinstance(v_pages, quantization_utils.QuantizedTensor):
    v_pages, v_scales_pages = v_pages.weight, v_pages.scales
    assert isinstance(v_scales_pages, jax.Array)  # For typing.
    v_scales_pages = jnp.broadcast_to(
        v_scales_pages, (*v_scales_pages.shape[:-1], v_pages.shape[-1])
    )
  else:
    v_scales_pages = None

  batch_size, query_len, num_q_heads, head_dim = q.shape
  # The reason why we reshape is that the 2nd last dim is query_len instead of
  # num_q_heads. 2nd last dim in kernel has to be a multiple of 8. num_heads
  # is hard to satisfy this requirement.
  q = jnp.permute_dims(q, (0, 2, 1, 3))
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"  # pytype: disable=attribute-error
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {lengths.dtype}"
    )
  if num_queries_per_compute_block > query_len:
    raise ValueError(f"{num_queries_per_compute_block=} should be smaller or equal to {query_len=}")
  if num_kv_pages_per_compute_block > pages_per_sequence:
    raise ValueError(f"{num_kv_pages_per_compute_block=} should be smaller or equal to {pages_per_sequence=}")
  if pages_per_sequence % num_kv_pages_per_compute_block != 0:
    raise ValueError(
        "num_kv_pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {pages_per_sequence=} and {num_kv_pages_per_compute_block=}."
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_q_heads} and {num_kv_heads}."
    )
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads

  # grid
  # query_len dim has to come before kv_len dim for the fa v1 implementation because if we do the other way around,
  # then for each j ([0, T_c]) and i ([0, T_r]), we load l_i and m_i from HBM and store to HBM.
  # then for j+1, we have to loop over i ([0, T_r]) which requires the load l_i and m_i.
  # But this is forbidden in Pallas: https://jax.readthedocs.io/en/latest/pallas/tpu/sparse.html#example-sparse-dense-matrix-multiplication
  # "When we change output block Pallas will finally store the output into HBM and assume we never touch it again."
  grid = (
      batch_size,
      num_kv_heads,
      # what if query_len%num_queries_per_compute_block!=0 or pages_per_sequence%num_kv_pages_per_compute_block!=0
      query_len // num_queries_per_compute_block,  # how many compute blocks we need to loop the query_len
      pages_per_sequence // num_kv_pages_per_compute_block,  # how many compute blocks we need to loop the kv_len
  )
  print(f'xw32 line471 {grid=}')

  # out_shape
  o_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  print(f'xw32 line520, after q.reshape, {q.shape=} {o_shape=}')
  l = jax.ShapeDtypeStruct(
      (batch_size, num_q_heads, query_len, MIN_BLOCK_SIZE), dtype=jnp.float32
  )
  m = jax.ShapeDtypeStruct(
      (batch_size, num_q_heads, query_len, MIN_BLOCK_SIZE), dtype=jnp.float32
  )
  k_debug = jax.ShapeDtypeStruct(
      (num_kv_pages_per_compute_block*page_size, head_dim), dtype=jnp.float32
  )  # k for given batch and kv_head, for debug only. Delete later.
  v_debug = jax.ShapeDtypeStruct(
      (num_kv_pages_per_compute_block*page_size, head_dim), dtype=jnp.float32
  )  # v for given batch and kv_head, for debug only. Delete later.
  q_debug = jax.ShapeDtypeStruct(
      (num_queries_per_compute_block, head_dim), dtype=jnp.float32
  )
  o_debug = jax.ShapeDtypeStruct(
      (batch_size, num_q_heads, query_len, head_dim), dtype=q.dtype
  )  # o_debug contains everything untrunked.
  out_shape = (o_shape, l, m, k_debug, v_debug, q_debug, o_debug)

  # in-spec. Note q.shape=[batch_size, num_q_heads, query_len, head_dim]
  # Map from grid idx.
  print(f'xw32 line591 {num_q_heads_per_kv_head=}')
  def qo_index_map(batch_index, kv_head_index, q_seq_blk_idx, *_):
    return (batch_index, kv_head_index, q_seq_blk_idx, 0)
  q_block_spec = pl.BlockSpec(
      # q_ref.shape=[1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
      (1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim),
      qo_index_map,  # map from grid idx to q's starting index
  )
  q_dtype_for_kernel_launch = q.dtype
  in_specs = [
      q_block_spec,
      # Below 4 correspond to the 4 input: k_pages, k_scales_pages, q_pages, q_scales_pages.
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      None,
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      None,
  ]

  # out_spec
  o_specs = pl.BlockSpec(  # Should be the same as q_block_spec
      # q_ref.shape=[1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
      (1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim),
      qo_index_map,  # map from grid idx to q's starting index
  )
  # lm_index_map is same as qo_index_map
  def lm_index_map(batch_index, kv_head_index, q_seq_blk_idx, *_):
    return (batch_index, kv_head_index, q_seq_blk_idx, 0)
  out_specs = [
    o_specs,
    pl.BlockSpec((1, num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
                 lm_index_map),  # l
    pl.BlockSpec((1, num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
                 lm_index_map),  # m
    pl.BlockSpec((num_kv_pages_per_compute_block*page_size, head_dim),
                 lambda batch_index, kv_head_index, q_seq_blk_idx, *_: (0, 0)),  # k_debug
    pl.BlockSpec((num_kv_pages_per_compute_block*page_size, head_dim),
                 lambda batch_index, kv_head_index, q_seq_blk_idx, *_: (0, 0)),  # v_debug
    pl.BlockSpec((num_queries_per_compute_block, head_dim),
                 lambda batch_index, kv_head_index, q_seq_blk_idx, *_: (0, 0)),  # q_debug
    pl.BlockSpec((batch_size, num_q_heads, query_len, head_dim),
                 lambda batch_index, kv_head_index, q_seq_blk_idx, *_: (0, 0, 0, 0)),  # o_debug
  ]

  # scratch space. Note k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
  # TODO(xiowei): one optimization is to check if
  # num_kv_pages_per_compute_block != pages_per_sequence per
  # http://google3/third_party/py/jax/experimental/pallas/ops/tpu/flash_attention.py;l=688;rcl=669291182
  # meaning if we don't tile the softmax, we can save a bunch of VPU ops.
  l_scratch = pltpu.VMEM((num_queries_per_compute_block, MIN_BLOCK_SIZE),
                          jnp.float32)
  m_scratch = pltpu.VMEM((num_queries_per_compute_block, MIN_BLOCK_SIZE),
                          jnp.float32)
  acc_scratch = pltpu.VMEM((num_queries_per_compute_block, head_dim),
                          jnp.float32)
  scratch_shapes = [
      pltpu.VMEM(
          (
              2,  # For double buffering during DMA copies.
              num_kv_pages_per_compute_block,
              page_size,
              head_dim,
          ),
          k_pages.dtype,
      ),  # k_pages buffer, k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
      None,  # k_scales_pages=None
      pltpu.VMEM(
          (
              2,  # For double buffering during DMA copies.
              num_kv_pages_per_compute_block,
              page_size,
              head_dim,
          ),
          v_pages.dtype,
      ),  # v_pages buffer
      None,  # v_scales_pages=None
      pltpu.SemaphoreType.DMA,
      l_scratch,
      m_scratch,
      acc_scratch,
  ]

  kernel = pl.pallas_call(
      functools.partial(
          paged_flash_attention_kernel,
          pages_per_sequence=pages_per_sequence,
          batch_size=batch_size,
          num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
          num_queries_per_compute_block=num_queries_per_compute_block,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          query_len=query_len
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          )
      ),
      out_shape=out_shape,
  )
  page_indices_1d = page_indices.reshape(-1)
  buffer_index = jnp.zeros((1,), jnp.int32)
  step = jnp.zeros((1,), jnp.int32)
  compiled_kernel = (
    jax.jit(kernel)
    .lower(
      # The first 4 are prefetched scalars.
      lengths,
      page_indices_1d,
      buffer_index,  # buffer index
      step,  # step
      q.astype(q_dtype_for_kernel_launch),
      k_pages,  # [num_kv_heads, total_num_pages, page_size, head_dim]
      k_scales_pages,
      v_pages,
      v_scales_pages,
    )
    .compile({'xla_tpu_enable_log_recorder': 'true'})
  )
  outs = compiled_kernel(
      # The first 4 are prefetched scalars.
      lengths,
      page_indices_1d,
      buffer_index,  # buffer index
      step,  # step
      q.astype(q_dtype_for_kernel_launch),
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
  )  # should get 3 return values.
  ret = outs[0]
  # print(f'xw32 finished the pallas kernel. {ret.shape=} Returning...', flush=True)
  # print(f'xw32 actual k={outs[3]}')
  # print(f'xw32 actual v={outs[4]}')
  # print(f'xw32 actual q={outs[5]}')
  # print(f'xw32 actual o_debug_ref.reshape(batch_size, query_len, num_q_heads, head_dim)={outs[6].reshape(batch_size, query_len, num_q_heads, head_dim)[:,:,1,:]}')
  # Reshape the output because we reshaped q at the beginning of the function.
  return jnp.permute_dims(ret, (0, 2, 1, 3)).astype(q.dtype)


