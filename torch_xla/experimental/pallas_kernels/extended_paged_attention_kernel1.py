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
from typing import Literal

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

def paged_flash_attention_kernel(
    # prefetched value
    lengths_ref, # [batch_size] jax.Array the length of each example
    page_indices_ref, # 1d vector, results from page_indices.reshape(-1) where originally page_indices.shape=[batch_size, pages_per_sequence]
    buffer_index_ref, # for double buffer
    step_ref, # xw32q: do we still need it?
    # input
    q_ref, # q_ref.shape=[num_queries_per_compute_block, head_dim]
    k_pages_hbm_ref, # shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    # output
    o_ref, # same shape as q_ref: [num_queries_per_compute_block, head_dim]
    m_ref,
    l_ref,
    # scratch space
    k_vmem_buffer, # shape=[2, num_kv_pages_per_compute_block, num_kv_heads, head_dim]
    k_scales_vmem_buffer,
    v_vmem_buffer, # shape=[2, num_kv_pages_per_compute_block, num_kv_heads, head_dim]
    v_scales_vmem_buffer,
    sem,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    batch_size: int,
    num_kv_pages_per_compute_block: int,
    num_queries_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
):
  """Pallas kernel for paged attention."""
  # grid=[batch_size, num_q_heads, num_q_len_blocks, num_kv_len_blocks]
  b, q_head_idx, q_blk_idx, kv_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
      pl.program_id(3),
  )

  num_kv_heads, total_num_pages, page_size, head_dim = k_pages_hbm_ref.shape
  assert q_ref.shape == [num_queries_per_compute_block, head_dim]
  compute_blk_size_kv = page_size * num_kv_pages_per_compute_block

  length_for_cur_batch = lengths_ref[b]
  
  def compute_block_indices(b, q_head_idx, kv_blk_idx):
    """Return next_b, next_q_head_idx, next_kv_blk_idx"""

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
          0,
          0,
      )

    def advance_h():
      next_h = h + 1
      return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

    return lax.cond(kv_blk_idx * compute_blk_size_kv < lengths_ref[b], lambda: (b, q_head_idx, kv_blk_idx), advance_h)

  def create_kv_async_copy_descriptors(b, kv_head_idx, i, buffer_index):
    page_offset = b * pages_per_sequence + i * num_kv_pages_per_compute_block
    pages_to_load = num_kv_pages_per_compute_block
    async_copy_k = MultiPageAsyncCopyDescriptor(
        k_pages_hbm_ref,
        k_scales_pages_hbm_ref,
        k_vmem_buffer.at[buffer_index],
        k_scales_vmem_buffer.at[buffer_index]
        if k_scales_vmem_buffer is not None
        else None,
        sem,
        page_indices_ref,
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

  @pl.when(kv_blk_idx == 0)
  def start_new_sequence():
    m_scratch_ref = jnp.full(m_scratch_ref.shape, -jnp.inf, jnp.float32)
    l_scratch_ref = jnp.zeros(l_scratch_ref.shape, jnp.float32)
    acc_scratch_ref = jnp.zeros(acc_scratch_ref.shape, jnp.float32)

  # xw32q: what if length % (i_kv_len * compute_blk_size_kv) != 0
  # for the current batch, query_head, and current kv_i block and q_i block, we do this
  # Flash attention begins
  m_prev = m_scratch_ref[...]
  l_prev = l_scratch_ref[...]
  q = q_ref[...].astype(jnp.float32)

  # Gather the k and v.
  step = step_ref[0]
  buffer_index = buffer_index_ref[0]
  kv_head_idx_to_fetch = q_head_idx // num_kv_heads

  @pl.when(step == 0)
  def prefetch_first_block():  # pylint: disable=unused-variable
    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        b, kv_head_idx_to_fetch, kv_blk_idx, buffer_index
    )
    async_copy_k.start()
    async_copy_v.start()
  
  next_b, next_q_head_idx, next_kv_blk_idx = compute_block_indices(b, q_head_idx, kv_blk_idx+1)

  @pl.when(next_b < batch_size)
  def prefetch_next_block():  # pylint: disable=unused-variable
    next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
    async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
        next_b, next_q_head_idx // num_kv_heads, next_kv_blk_idx, next_buffer_index
    )
    async_copy_next_k.start()
    async_copy_next_v.start()
    buffer_index_ref[0] = next_buffer_index
  
  async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
    b, kv_head_idx_to_fetch, kv_blk_idx, buffer_index 
  )
  k = async_copy_k.wait_and_get_loaded()
  # TODO: continue to do attention.



MIN_BLOCK_SIZE = 1

@functools.partial(
    jax.jit,
    static_argnames=[
        "num_kv_pages_per_compute_block",
        "attn_logits_soft_cap",
        "mask_value",
        "megacore_mode",
        "inline_seq_dim",
    ],
)
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
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"  # pytype: disable=attribute-error
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_q_heads} and {num_kv_heads}."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if pages_per_sequence % num_kv_pages_per_compute_block != 0:
    raise ValueError(
        "num_kv_pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {num_kv_pages_per_compute_block} and {pages_per_sequence}."
    )
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        "The dtype of `lengths` must be int32. Got {lengths.dtype}"
    )
  if (num_q_heads // num_kv_heads) % 8 != 0:
    # xw32: do we need to support this case? FWIW, FA pallas kernel only support num_q_heads == num_kv_heads 
    # The original paged_attention kernel support it but it doesn't provide much value.
    raise ValueError('(num_q_heads // num_kv_heads) % 8 != 0')

  # Here, we guarantee (num_q_heads // num_kv_heads) % 8 == 0
  # grid
  grid = (
      batch_size,
      num_q_heads,
      # what if query_len%num_queries_per_compute_block!=0 or pages_per_sequence%num_kv_pages_per_compute_block!=0
      query_len // num_queries_per_compute_block, # how many compute blocks we need to loop the query_len
      pages_per_sequence // num_kv_pages_per_compute_block, # how many compute blocks we need to loop the kv_len
  )  # type: ignore

  # out_shape
  o_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  # xw32 what should be the 4th dimension? In FA, it's MIN_BLOCK_SIZE
  # In PA, it's 1.
  # TODO(xw32): change the 4th dimension of l_shape and m_shape to something else later.
  l_shape = jax.ShapeDtypeStruct(
    (batch_size, query_len, num_q_heads, MIN_BLOCK_SIZE), dtype=jnp.float32
  )
  m_shape = jax.ShapeDtypeStruct(
    (batch_size, query_len, num_q_heads, MIN_BLOCK_SIZE), dtype=jnp.float32
  )
  out_shape = [o_shape, l_shape, m_shape]

  # out_spec
  o_specs = pl.BlockSpec( # same as q_block_spec
      (None, num_queries_per_compute_block, None, head_dim), # q_ref.shape=[num_queries_per_compute_block, head_dim]
      lambda batch_index, head_index, q_seq_blk_idx, *_: (batch_index, q_seq_blk_idx, head_index, 0), # map from grid idx to q's starting index
  )
  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, q_seq_index, head_index, 0)
  l_specs = pl.BlockSpec(
    (None, num_queries_per_compute_block, None, MIN_BLOCK_SIZE), lm_index_map
  )
  m_specs = pl.BlockSpec(
    (None, num_queries_per_compute_block, None, MIN_BLOCK_SIZE), lm_index_map
  )
  out_specs = [o_specs, l_specs, m_specs]

  # in-spec and scratch space. Note q.shape=[batch_size, query_len, num_q_heads, head_dim]
  q_block_spec = pl.BlockSpec(
      (None, num_queries_per_compute_block, None, head_dim), # q_ref.shape=[num_queries_per_compute_block, head_dim]
      lambda batch_index, head_index, q_seq_blk_idx, *_: (batch_index, q_seq_blk_idx, head_index, 0), # map from grid idx to q's starting index
  )
  q_dtype_for_kernel_launch = q.dtype
  if k_scales_pages is not None and v_scales_pages is not None:
    # TODO(xw32): do it later when we need to handle quantization
    assert False, 'shouldnt run here'
    in_specs = [
        q_block_spec,
        # pltpu.TPUMemorySpace.ANY means we are putting everything in HBM.
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
    ]
    scratch_shapes = (
        # xw32: how is the pltpu.VMEM being used? I see. It's used in the kernel.
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                num_kv_pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),  # k_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                num_kv_pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_scales_pages.dtype,  # pytype: disable=attribute-error
        ),  # k_scales_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                num_kv_pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),  # v_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                num_kv_pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_scales_pages.dtype,  # pytype: disable=attribute-error
        ),  # v_scales_pages buffer
        pltpu.SemaphoreType.DMA,
    )
  else: # Non-quantization branch. either k_scales_pages or v_scales_pages is None.
    in_specs = [
        q_block_spec,
        # Below 4 correspond to the 4 input: k_pages, k_scales_pages, q_pages, q_scales_pages.
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        None,  # type: ignore[list-item]  k_scales_pages=None
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        None,  # type: ignore[list-item]  v_scales_pages=None
    ]
    # Note k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
    scratch_shapes = (
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                num_kv_pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),  # k_pages buffer, k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
        None, # k_scales_pages=None
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                num_kv_pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),  # v_pages buffer
        None, # v_scales_pages=None
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((num_queries_per_compute_block, MIN_BLOCK_SIZE), jnp.float32), # m_scratch
        pltpu.VMEM((num_queries_per_compute_block, MIN_BLOCK_SIZE), jnp.float32), # l_scratch
        pltpu.VMEM((num_queries_per_compute_block, head_dim), jnp.float32), # acc_scratch
    )

  out, _, _ = pl.pallas_call(
      functools.partial(
          paged_flash_attention_kernel,
          pages_per_sequence=pages_per_sequence,
          batch_size=batch_size,
          num_kv_pages_per_compute_block=num_kv_pages_per_compute_block,
          num_queries_per_compute_block=num_queries_per_compute_block,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          # There are 4 scalars prefetched per kernel call: `lengths_ref`,
          # `page_indices_ref`, `buffer_index_ref`, `step_ref`
          num_scalar_prefetch=4,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      # compiler_params=pltpu.TPUCompilerParams(
      #     dimension_semantics=dimension_semantics), # do we need it?
      out_shape=out_shape,
  )(
      # The first 4 are prefetched scalars.
      lengths,
      page_indices.reshape(-1),
      jnp.zeros((1,), jnp.int32),  # buffer index
      jnp.zeros((1,), jnp.int32),  # step
      q.astype(q_dtype_for_kernel_launch),
      k_pages,
      k_scales_pages,
      v_pages,
      v_scales_pages,
  )
  print(f'xw32 finished the pallas kernel. {out.shape=} Returning...', flush=True)
  return out.reshape(batch_size, query_len, num_q_heads, head_dim).astype(q.dtype)


