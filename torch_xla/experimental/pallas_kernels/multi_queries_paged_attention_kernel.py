"""PagedAttention TPU kernel with query_len>1 support."""

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
      pages_hbm_ref,  # [num_kv_heads, total_num_pages, page_size, head_dim]
      scales_pages_hbm_ref,
      vmem_buffer,  # [pages_per_compute_block, page_size, head_dim]
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
    if (self._scales_pages_hbm_ref is not None and
        self._scales_vmem_buffer is not None):
      self._async_copies += [
          self._make_scales_async_copy(i)
          for i in range(self._num_pages_to_load)
      ]

  def _make_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    return pltpu.make_async_copy(self._pages_hbm_ref.at[page_index],
                                 self._vmem_buffer.at[i], self._sem)

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
    # Return value shape is [pages_per_compute_block*page_size, head_dim]
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


def _flash_attention(
    q_head_idx_per_kv,  # scalar, ranges from 0 to num_query_heads_per_kv_head
    lengths_ref,  # [batch_size] jax.Array the length of each example
    effective_q_lens_ref,  # [batch_size] jax.Array the length of the effective query lengths
    # input
    q_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    k,  # [pages_per_compute_block*page_size,head_dim]
    v,  # [pages_per_compute_block*page_size,head_dim]
    # output
    o_ref,  # same shape as q_ref: [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    l_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE]
    m_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE]
    l_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE]
    m_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE]
    acc_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    *,
    num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int,
    mask_value: float,
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

  @pl.when(kv_blk_idx == 0)
  def start_new_sequence():
    l_scratch_ref[q_head_idx_per_kv] = jnp.zeros(
        l_scratch_ref[q_head_idx_per_kv].shape, jnp.float32)
    m_scratch_ref[q_head_idx_per_kv] = jnp.full(
        m_scratch_ref[q_head_idx_per_kv].shape, -jnp.inf, jnp.float32)
    acc_scratch_ref[q_head_idx_per_kv] = jnp.zeros(
        acc_scratch_ref[q_head_idx_per_kv].shape, jnp.float32)

  m_prev = m_scratch_ref[q_head_idx_per_kv]
  l_prev = l_scratch_ref[q_head_idx_per_kv]
  q = q_ref[0,
            q_head_idx_per_kv, :, :].astype(jnp.float32)  # [block_q, head_dim]
  assert q.shape == (num_queries_per_compute_block, head_dim)
  kv_seq_len_per_kv_compute_blk = num_kv_pages_per_compute_block * page_size
  assert k.shape == (kv_seq_len_per_kv_compute_blk, head_dim)
  s = jnp.einsum(
      'qd,td->qt', q, k,
      preferred_element_type=jnp.float32)  # [block_q, block_k]
  assert s.shape == (num_queries_per_compute_block,
                     kv_seq_len_per_kv_compute_blk)

  q_index = q_blk_idx * num_queries_per_compute_block
  kv_index = kv_blk_idx * kv_seq_len_per_kv_compute_blk
  effective_kv_len = lengths_ref[b]
  effective_q_len = effective_q_lens_ref[b]
  row_ids = (
      effective_kv_len - effective_q_len) + q_index + jax.lax.broadcasted_iota(
          jnp.int32,
          (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk), 0)
  col_ids = kv_index + jax.lax.broadcasted_iota(
      jnp.int32,
      (num_queries_per_compute_block, kv_seq_len_per_kv_compute_blk), 1)
  causal_mask = jnp.where(row_ids < col_ids, mask_value, 0.)
  assert causal_mask.shape == (num_queries_per_compute_block,
                               kv_seq_len_per_kv_compute_blk)
  s = s + causal_mask  # [block_q, block_k]
  assert s.shape == (num_queries_per_compute_block,
                     kv_seq_len_per_kv_compute_blk)

  m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
  m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

  block_k_repeats, rem = divmod(kv_seq_len_per_kv_compute_blk, MIN_BLOCK_SIZE)
  if rem:
    raise NotImplementedError(
        f"{kv_seq_len_per_kv_compute_blk=} should be a multiple of {MIN_BLOCK_SIZE}"
    )
  p = jnp.exp(
      s - pltpu.repeat(m_next, block_k_repeats, 1))  # Shape [block_q, block_k]

  alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128]

  l_corr = alpha * l_prev  # Shape [block_q, 128]

  l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

  head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
  l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
  if rem:
    if head_dim_repeats == 0:
      l_broadcast = lambda l: l[:, :head_dim]
    else:
      raise NotImplementedError(
          f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
  # Need to store these l_next and m_next which will relay to the output.
  l_scratch_ref[q_head_idx_per_kv] = l_next
  m_scratch_ref[q_head_idx_per_kv] = m_next

  l_next_inv_safe = jnp.where(l_next == 0.0, 1.0,
                              1.0 / l_next)  # [block_q, 128]

  acc_scratch_ref[q_head_idx_per_kv] *= l_broadcast(l_corr * l_next_inv_safe)
  # Note Matmul operandlhs must have a shape divisible by (16, 1)
  o_curr = jax.lax.dot(
      p.astype(v.dtype), v,
      preferred_element_type=jnp.float32)  # [block_q, 128]

  acc_scratch_ref[q_head_idx_per_kv] += o_curr * l_broadcast(l_next_inv_safe)

  # The condition comes from the check controlling if we should run the function get_kv_and_run_flash_attention.
  # If kv_len=512, kv_seq_len_per_kv_compute_blk=256, then last kv_blk_idx that we need to store_to_output is 1.
  # If kv_len=513, kv_seq_len_per_kv_compute_blk=256, then last kv_blk_idx that we need to store_to_output is 2.
  is_last_kv_blk_idx = kv_blk_idx == pl.cdiv(effective_kv_len,
                                             kv_seq_len_per_kv_compute_blk) - 1
  is_next_kv_blk_masked_out = jnp.logical_not(
      _block_below_or_on_diag(q_blk_idx, num_queries_per_compute_block,
                              kv_blk_idx + 1, kv_seq_len_per_kv_compute_blk,
                              effective_q_len, effective_kv_len))

  @pl.when(jnp.logical_or(is_last_kv_blk_idx, is_next_kv_blk_masked_out))
  def store_to_output():
    o_ref[0, q_head_idx_per_kv] = acc_scratch_ref[q_head_idx_per_kv].astype(
        o_ref.dtype)
    l_ref[0, q_head_idx_per_kv] = l_scratch_ref[q_head_idx_per_kv].astype(
        l_ref.dtype)
    m_ref[0, q_head_idx_per_kv] = m_scratch_ref[q_head_idx_per_kv].astype(
        m_ref.dtype)


# A block is considered below or on diagonal as long as the bottom left
# corner of the block is below or on diagonal.
# If the inputs are 0, 32, 0, 256, 64, 257, the block's bottom left corner is (31, 0). For that column(0), the diagonal element is (-193, 0). We check(>=) the x-coordinate of the corner and the diagonal element (31 and -193)
# If the inputs are 0, 32, 1, 256, 64, 257, the block's bottom left corner is (31, 256). For that column(256), the diagonal element is (63, 256). We check(>=) the x-coordinate of the corner and the diagonal element (31 and 63).
def _block_below_or_on_diag(q_blk_idx, q_blk_size, kv_blk_idx, kv_blk_size,
                            effective_q_len, effective_kv_len):
  return ((q_blk_idx + 1) * q_blk_size - 1) >= (kv_blk_idx * kv_blk_size) - (
      effective_kv_len - effective_q_len)


def paged_flash_attention_kernel(
    lengths_ref,  # [batch_size] jax.Array the length of each example
    # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[batch_size, pages_per_sequence]
    page_indices_ref,
    effective_q_lens_ref,  # [batch_size] jax.Array the length of the effective query lengths
    buffer_index_ref,
    step_ref,
    # At caller, q.shape=[batch_size, num_q_heads query_len, head_dim]
    q_ref,  # q_ref.shape=[1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    k_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,  # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    v_scales_pages_hbm_ref,
    # same shape as q_ref: [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim], output
    o_ref,
    l_ref,
    m_ref,
    # scratch space
    k_vmem_buffer,  # (2, num_kv_pages_per_compute_block, num_kv_heads, head_dim)
    k_scales_vmem_buffer,
    v_vmem_buffer,  # (2, num_kv_pages_per_compute_block, num_kv_heads, head_dim)
    v_scales_vmem_buffer,
    sem,
    l_scratch_ref,
    m_scratch_ref,
    acc_scratch_ref,
    *,
    pages_per_sequence: int,  # Note [bs, pages_per_sequence] = page_indices.shape
    batch_size: int,
    num_kv_pages_per_compute_block: int,
    mask_value: float,
    query_len: int,
):
  """Pallas kernel for paged attention."""
  b, kv_head_idx, q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
      pl.program_id(3),
  )
  num_q_blks = pl.num_programs(2)
  b_q_ref, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim = q_ref.shape
  num_kv_heads, total_num_pages, page_size, head_dim = k_pages_hbm_ref.shape
  compute_blk_size_kv = page_size * num_kv_pages_per_compute_block
  effective_kv_len = lengths_ref[b]
  effective_q_len = effective_q_lens_ref[b]

  # Get the K and V for the current batch and current kv head.
  should_run = jnp.logical_and(
      kv_blk_idx * compute_blk_size_kv < effective_kv_len,
      _block_below_or_on_diag(q_blk_idx, num_queries_per_compute_block,
                              kv_blk_idx, compute_blk_size_kv, effective_q_len,
                              effective_kv_len))

  @pl.when(should_run)
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
        # assumption: kv_blk_idx * compute_blk_size_kv >= lengths_ref[b], or the block is above the causal mask diagonal.
        next_kv_head_idx = kv_head_idx + 1
        return lax.cond(
            q_blk_idx == num_q_blks - 1,
            lambda: lax.cond(next_kv_head_idx < num_kv_heads, lambda:
                             (b, next_kv_head_idx, 0), advance_b), lambda:
            (b, kv_head_idx, 0))

      return lax.cond(
          jnp.logical_and(
              kv_blk_idx * compute_blk_size_kv < lengths_ref[b],
              _block_below_or_on_diag(q_blk_idx, num_queries_per_compute_block,
                                      kv_blk_idx, compute_blk_size_kv,
                                      effective_q_lens_ref[b], lengths_ref[b])),
          lambda: (b, kv_head_idx, kv_blk_idx), advance_kv_head_idx)

    def create_kv_async_copy_descriptors(b, kv_head_idx, kv_blk_idx,
                                         buffer_index):
      page_offset = b * pages_per_sequence + kv_blk_idx * num_kv_pages_per_compute_block
      pages_to_load = num_kv_pages_per_compute_block
      async_copy_k = MultiPageAsyncCopyDescriptor(
          k_pages_hbm_ref,
          k_scales_pages_hbm_ref,
          k_vmem_buffer.at[buffer_index],
          k_scales_vmem_buffer.at[buffer_index]
          if k_scales_vmem_buffer is not None else None,
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
          if v_scales_vmem_buffer is not None else None,
          sem,
          page_indices_ref,
          page_offset,
          pages_to_load,
          kv_head_idx,
      )
      return async_copy_k, async_copy_v

    step = step_ref[0]
    buffer_index = buffer_index_ref[0]

    @pl.when(step == 0)
    def prefetch_first_block():  # pylint: disable=unused-variable
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          b, kv_head_idx, kv_blk_idx, buffer_index)
      async_copy_k.start()
      async_copy_v.start()

    next_b, next_kv_head_idx, next_kv_blk_idx = compute_block_indices(
        b, kv_head_idx, q_blk_idx, kv_blk_idx + 1)

    @pl.when(next_b < batch_size)
    def prefetch_next_block():  # pylint: disable=unused-variable
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_b, next_kv_head_idx, next_kv_blk_idx, next_buffer_index)
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index

    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        b, kv_head_idx, kv_blk_idx, buffer_index)
    k = async_copy_k.wait_and_get_loaded(
    )  # [pages_per_compute_block*page_size,head_dim]
    v = async_copy_v.wait_and_get_loaded()

    for q_head_idx in range(num_q_heads_per_kv_head):
      _flash_attention(
          q_head_idx,
          lengths_ref,
          effective_q_lens_ref,
          q_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
          k,
          v,
          o_ref,  # [1, num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
          l_ref,
          m_ref,
          l_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE]
          m_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE]
          acc_scratch_ref,  # [num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
          num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
          num_queries_per_compute_block=num_queries_per_compute_block,
          mask_value=mask_value,
          query_len=query_len,
          page_size=page_size,
          head_dim=head_dim,
      )
    # o_ref.shape=[num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim]
    step_ref[0] = step + 1

  # end of get_kv_and_run_flash_attention.


MIN_BLOCK_SIZE = 128


@jax.profiler.annotate_function
@functools.partial(
    jax.jit,
    static_argnames=[
        "num_kv_pages_per_compute_block",
        "num_queries_per_compute_block",
        "mask_value",
    ],
)
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array | quantization_utils.QuantizedTensor,
    v_pages: jax.Array | quantization_utils.QuantizedTensor,
    lengths: jax.Array,
    page_indices: jax.Array,
    effective_q_lens: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int = 4,
) -> jax.Array:
  """Paged grouped query attention.

  Args:
    q: A [batch_size, query_len, num_q_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array the effective kv length of each example.
    page_indices: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    effective_q_lens: A i32[batch_size] jax.Array the effective query length of each example.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    num_kv_pages_per_compute_block: how many kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_compute_block: how many queries to be processes in one flash attention block in the pallas kernel.

  Returns:
    The output of attention([batch_size, query_len, num_q_heads, head_dim]).
  """
  if isinstance(k_pages, quantization_utils.QuantizedTensor):
    k_pages, k_scales_pages = k_pages.weight, k_pages.scales
    assert isinstance(k_scales_pages, jax.Array)  # For typing.
    k_scales_pages = jnp.broadcast_to(
        k_scales_pages, (*k_scales_pages.shape[:-1], k_pages.shape[-1]))
  else:
    k_scales_pages = None
  if isinstance(v_pages, quantization_utils.QuantizedTensor):
    v_pages, v_scales_pages = v_pages.weight, v_pages.scales
    assert isinstance(v_scales_pages, jax.Array)  # For typing.
    v_scales_pages = jnp.broadcast_to(
        v_scales_pages, (*v_scales_pages.shape[:-1], v_pages.shape[-1]))
  else:
    v_scales_pages = None

  batch_size, query_len, num_q_heads, head_dim = q.shape
  q = jnp.permute_dims(q, (0, 2, 1, 3))
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"  # pytype: disable=attribute-error
    )
  if head_dim_k != head_dim:
    raise ValueError("head_dim of Q must be the same as that of K/V. Got"
                     f" {head_dim} and {head_dim_k}.")
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if lengths.shape != effective_q_lens.shape:
    raise ValueError(
        "`lengths` and `effective_q_lens` must have the same size: batch_size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {lengths.dtype}")
  if num_queries_per_compute_block > query_len:
    raise ValueError(
        f"{num_queries_per_compute_block=} should be smaller or equal to {query_len=}"
    )
  if num_kv_pages_per_compute_block > pages_per_sequence:
    raise ValueError(
        f"{num_kv_pages_per_compute_block=} should be smaller or equal to {pages_per_sequence=}"
    )
  if pages_per_sequence % num_kv_pages_per_compute_block != 0:
    raise ValueError(
        "num_kv_pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {pages_per_sequence=} and {num_kv_pages_per_compute_block=}.")
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_q_heads} and {num_kv_heads}.")
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads

  # grid
  grid = (
      batch_size,
      num_kv_heads,
      pl.cdiv(query_len, num_queries_per_compute_block
             ),  # how many compute blocks we need to loop the query_len
      pages_per_sequence //
      num_kv_pages_per_compute_block,  # how many compute blocks we need to loop the kv_len
  )

  # out_shape
  o_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  l = jax.ShapeDtypeStruct((batch_size, num_q_heads, query_len, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  m = jax.ShapeDtypeStruct((batch_size, num_q_heads, query_len, MIN_BLOCK_SIZE),
                           dtype=jnp.float32)
  out_shape = (o_shape, l, m)

  # in-spec. Note q.shape=[batch_size, num_q_heads, query_len, head_dim]
  # Map from grid idx.

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
      pl.BlockSpec((1, num_q_heads_per_kv_head, num_queries_per_compute_block,
                    MIN_BLOCK_SIZE), lm_index_map),  # l
      pl.BlockSpec((1, num_q_heads_per_kv_head, num_queries_per_compute_block,
                    MIN_BLOCK_SIZE), lm_index_map),  # m
  ]

  # scratch space. Note k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
  l_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
      jnp.float32)
  m_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_compute_block, MIN_BLOCK_SIZE),
      jnp.float32)
  acc_scratch = pltpu.VMEM(
      (num_q_heads_per_kv_head, num_queries_per_compute_block, head_dim),
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
          mask_value=mask_value,
          query_len=query_len),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=5,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.TPUCompilerParams(
          # due to compute_block_indices, we loop batch, kv_head, q_blk, kv_blk, the order matters.
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
              "arbitrary",
              "arbitrary",
          )),
      out_shape=out_shape,
  )
  page_indices_1d = page_indices.reshape(-1)
  buffer_index = jnp.zeros((1,), jnp.int32)
  step = jnp.zeros((1,), jnp.int32)
  outs = kernel(
      lengths,
      page_indices_1d,
      effective_q_lens,
      buffer_index,
      step,
      q.astype(q_dtype_for_kernel_launch),
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
  )
  ret = outs[0]
  return jnp.permute_dims(ret, (0, 2, 1, 3)).astype(q.dtype)
