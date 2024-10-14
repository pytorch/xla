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
      pages_hbm_ref,
      scales_pages_hbm_ref,
      vmem_buffer,
      scales_vmem_buffer,
      sem,
      page_indices,
      page_indices_start_offset,
      num_pages_to_load,
      head_index,
  ):
    # Original k_pages has shape [num_kv_heads, total_num_pages, page_size, head_dim]
    self._vmem_buffer = vmem_buffer
    self._scales_vmem_buffer = scales_vmem_buffer
    self._num_pages_to_load = num_pages_to_load
    if head_index is not None:
      self._pages_hbm_ref = pages_hbm_ref.at[head_index]
      if scales_pages_hbm_ref is not None:
        self._scales_pages_hbm_ref = scales_pages_hbm_ref.at[head_index]
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
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    q_ref,
    k_pages_hbm_ref,
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    k_scales_vmem_buffer,
    v_vmem_buffer,
    v_scales_vmem_buffer,
    sem,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    queries_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    program_ids=(),
):
  """Pallas kernel for paged attention."""
  # Originally if inline_seq_dim if False, grid=[num_cores, batch_size, num_heads, num_kv_len_blocks, num_queries_len_blocks]
  if program_ids: # inline_seq_dim
    core_index, b, h, i_kv_len, i_q_len = program_ids
  else: # inline_seq_dim==False
    core_index, b, h, i_kv_len, i_q_len = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
        pl.program_id(4),
    )

  num_kv_heads, _, page_size, head_dim = k_pages_hbm_ref.shape
  assert q_ref.shape == [queries_per_compute_block, head_dim]
  bk_kv = page_size * pages_per_compute_block

  b_step = 1
  b_start = 0
  



def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    q_ref,
    k_pages_hbm_ref,
    k_scales_pages_hbm_ref,
    v_pages_hbm_ref,
    v_scales_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    k_scales_vmem_buffer,
    v_vmem_buffer,
    v_scales_vmem_buffer,
    sem,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
):
  ...

@functools.partial(
    jax.jit,
    static_argnames=[
        "pages_per_compute_block",
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
    pages_per_compute_block: int,
    queries_per_compute_block: int = 4,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> jax.Array:
  """Paged grouped query attention.

  Args:
    q: A [batch_size, query_len, num_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array the length of each example.
    page_indices: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    attn_logits_soft_cap: The value used for soft capping the attention logits.
    pages_per_compute_block: how many pages to be processed in one flash
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
    The output of attention([batch_size, query_len, num_heads, head_dim]).
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

  # TODO(xw32): consider renaming num_heads to num_query_heads
  batch_size, query_len, num_heads, head_dim = q.shape
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"  # pytype: disable=attribute-error
    )
  if num_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_heads} and {num_kv_heads}."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if pages_per_sequence % pages_per_compute_block != 0:
    raise ValueError(
        "pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {pages_per_compute_block} and {pages_per_sequence}."
    )
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        "The dtype of `lengths` must be int32. Got {lengths.dtype}"
    )
  
    # TODO(dinghua): get the actual cores per chip once there's an official API.
  if megacore_mode == "kv_head":
    if num_kv_heads % 2 != 0:
      raise ValueError(
          "number of KV heads must be even when megacore_mode is 'kv_head'"
      )
    num_cores = 2
  elif megacore_mode == "batch":
    if batch_size % 2 != 0:
      raise ValueError("batch size must be even when megacore_mode is 'batch'")
    num_cores = 2
  elif megacore_mode is None:
    num_cores = 1
  else:
    raise ValueError("megacore_mode must be one of ['kv_head', 'batch', None]")

  if (num_heads // num_kv_heads) % 8 != 0:
    # TODO(xw32):add the query_len dim to this branch later.
    # Reshape q to hint XLA to pick a <1x128> layout otherwise it will pick a
    # <8x128> layout for a <1x128> memref inside the kernel and error out.
    q = q.reshape(batch_size, num_heads, 1, head_dim)
    if megacore_mode == "kv_head":
      q_block_spec = pl.BlockSpec(
          (None, num_heads // num_kv_heads, None, head_dim),
          lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0, 0),
      )
    elif megacore_mode == "batch":
      q_block_spec = pl.BlockSpec(
          (None, num_heads // num_kv_heads, None, head_dim),
          lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0, 0),
      )
    else:
      q_block_spec = pl.BlockSpec(
          (None, num_heads // num_kv_heads, None, head_dim),
          lambda core_index, b, h, *_: (b, h, 0, 0),
      )
    q_dtype_for_kernel_launch = jnp.float32
  else:
    if megacore_mode == "kv_head":
      # q.shape=[batch_size, query_len, num_heads, head_dim]
      q_block_spec = pl.BlockSpec(
          (None, None, num_heads // num_kv_heads, head_dim),
          lambda core_index, q, b, h, *_: (b, q, h * num_cores + core_index, 0),
      )
    elif megacore_mode == "batch":
      q_block_spec = pl.BlockSpec(
          (None, None, num_heads // num_kv_heads, head_dim),
          lambda core_index, q, b, h, *_: (b * num_cores + core_index, q, h, 0),
      )
    else: # (num_heads // num_kv_heads) % 8 == 0 and megacore_mode is None
      # q.shape=[batch_size, query_len, num_heads, head_dim]
      # if inline_seq_dim if False, grid=[num_cores, batch_size, num_heads, num_kv_len_blocks, num_queries_len_blocks]
      q_block_spec = pl.BlockSpec(
          (None, queries_per_compute_block, None, head_dim), # q block shape
          lambda core_index, b, h, i_kv_len, i_q_len, *_: (b, i_q_len, h, 0),
      )
    q_dtype_for_kernel_launch = q.dtype
  
  dimension_semantics: Sequence[Literal["parallel", "arbitrary"]]
  if inline_seq_dim:
    # xw32: ignore this branch for now.
    kernel = paged_flash_attention_kernel_inline_seq_dim
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_heads,
    )
    # xw32q: shouldn't batch dim and kv_heads dim be parallel?
    # both batch dim and kv heads are independent and can be parallel.
    dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary")
  else:
    print('xw32 line547, inline_seq_dim is False', flush=True)
    kernel = paged_flash_attention_kernel
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_heads,
        pages_per_sequence // pages_per_compute_block, # how many compute blocks we need to loop the kv_len
        query_len // queries_per_compute_block, # how many compute blocks we need to loop the query_len
    )  # type: ignore
    dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary", "arbitrary")
  
  if k_scales_pages is not None and v_scales_pages is not None:
    # TODO(xw32): do it later when we need to handle quantization
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
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),  # k_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_scales_pages.dtype,  # pytype: disable=attribute-error
        ),  # k_scales_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),  # v_pages buffer
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_scales_pages.dtype,  # pytype: disable=attribute-error
        ),  # v_scales_pages buffer
        pltpu.SemaphoreType.DMA,
    )
  else: # either k_scales_pages or v_scales_pages is None.
    in_specs = [
        q_block_spec,
        # Below 4 correspond to the 4 input: k_pages, k_scales_pages, q_pages, etc.
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        None,  # type: ignore[list-item]  k_scales_pages=None
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        None,  # type: ignore[list-item]  v_scales_pages=None
    ]
    scratch_shapes = (
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            k_pages.dtype,
        ),  # k_pages buffer, k_pages.shape=[num_kv_heads, total_num_pages, page_size, head_dim]
        None, # k_scales_pages=None
        pltpu.VMEM(
            (
                2,  # For double buffering during DMA copies.
                pages_per_compute_block,
                page_size,
                head_dim,
            ),
            v_pages.dtype,
        ),  # v_pages buffer
        None, # v_scales_pages=None
        pltpu.SemaphoreType.DMA,
    )

  out, _, _ = pl.pallas_call(
      functools.partial(
          kernel,
          pages_per_sequence=pages_per_sequence,
          batch_size=batch_size,
          pages_per_compute_block=pages_per_compute_block,
          queries_per_compute_block=queries_per_compute_block,
          mask_value=mask_value,
          attn_logits_soft_cap=attn_logits_soft_cap,
          megacore_mode=megacore_mode,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          # There are 4 scalars prefetched per kernel call: `lengths_ref`,
          # `page_indices_ref`, `buffer_index_ref`, `step_ref`
          num_scalar_prefetch=4,
          in_specs=in_specs,
          out_specs=[
              q_block_spec,
              q_block_spec,
              q_block_spec,
          ],
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=dimension_semantics),
      out_shape=[
          jax.ShapeDtypeStruct(q.shape, q_dtype_for_kernel_launch),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
      ],
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
  print('xw32 finished the pallas kernel. Returning...', flush=True)
  return out.reshape(batch_size, query_len, num_heads, head_dim).astype(q.dtype)


