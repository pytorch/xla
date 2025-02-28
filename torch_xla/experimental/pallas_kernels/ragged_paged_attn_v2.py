import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[num_seqs, pages_per_seq]
      offset,  # [seq_idx, kv_pages_start]
  ):
    self._vmem_buf = vmem_buf
    seq_id, kv_pages_start = offset
    self._async_copies = [
        pltpu.make_async_copy(
            pages_hbm_ref.at[page_indices_ref[seq_id, kv_pages_start + i]],
            vmem_buf.at[i],
            sem,
        )
        for i in range(vmem_buf.shape[0])
    ]

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait(self):
    for async_copy in self._async_copies:
      async_copy.wait()
    return self._vmem_buf


# TODO(jevinjiang): Move cur_q_len and cur_kv_len check to Pytorch and check on
# runtime! Too expensive for JIT.
def check_kernel_input(
    q,  # [total_num_tokens, num_q_heads, head_dim]
    k_pages,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens,  # i32[num_seqs]
    page_indices,  # i32[num_seqs, pages_per_seq]
    cu_q_lens,  # i32[num_seqs + 1]
    num_seqs,  # i32
    num_kv_pages_per_blk,  # i32
):
  num_tokens, num_q_heads, head_dim = q.shape
  _, page_size, num_kv_heads, head_dim_k = k_pages.shape
  _, pages_per_seq = page_indices.shape
  # TODO(jevinjiang): check heads are power of 2!
  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"{k_pages.shape=} and {v_pages.shape=} must have the same shape."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if kv_lens.shape[0] != num_seqs:
    raise ValueError(f"{kv_lens.shape[0]=} must be the same as {num_seqs=}")
  if page_indices.shape[0] != num_seqs:
    raise ValueError(
        f"{page_indices.shape[0]=} must be the same as {num_seqs=}"
    )
  if cu_q_lens.shape[0] != num_seqs + 1:
    raise ValueError(
        f"{cu_q_lens.shape[0]=} must be the same as {(num_seqs + 1)=}"
    )
  if num_seqs > num_tokens:
    raise ValueError(f"{num_seqs=} must be less or equal to {num_tokens=}")
  if (
      kv_lens.dtype != jnp.int32
      or page_indices.dtype != jnp.int32
      or cu_q_lens.dtype != jnp.int32
  ):
    raise ValueError(
        f"The dtype of `lengths` must be int32. Got {kv_lens.dtype=}, "
        f"{page_indices.dtype=}, {cu_q_lens.dtype=}"
    )
  if num_kv_pages_per_blk > pages_per_seq:
    raise ValueError(
        f"{num_kv_pages_per_blk=} should be smaller or equal to"
        f" {pages_per_seq=}"
    )
  if pages_per_seq % num_kv_pages_per_blk != 0:
    raise ValueError(
        f"{pages_per_seq=} must be divisible by {num_kv_pages_per_blk=}"
    )
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
  if num_kv_pages_per_blk * page_size % 128 != 0:
    raise ValueError(
        f"{num_kv_pages_per_blk=} * {page_size=} must be divisible by 128"
    )


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [num_seqs]
    page_indices_ref,  # [num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [num_seqs + 1]
    seq_buf_idx_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    k_pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    k_bufs,  # [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
    v_bufs,  # [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    *,
    mask_value: float,
    sm_scale: float,
):
  num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
  num_seqs = kv_lens_ref.shape[0]
  _, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, _ = k_bufs.shape
  num_kv_per_blk = num_kv_pages_per_blk * page_size
  num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk

  heads_blk_idx, q_blk_idx = (
      pl.program_id(0),
      pl.program_id(1),
  )
  num_heads_blks = pl.num_programs(0)
  init_seq_idx = seq_buf_idx_ref[0]
  init_buf_idx = seq_buf_idx_ref[1]

  q_len_start = q_blk_idx * num_q_per_blk
  q_len_end = q_len_start + num_q_per_blk

  # pl.debug_print("[jevin debug] -----------New loop in Pipeline------------")
  # pl.debug_print("[jevin debug] heads_blk_idx={}", heads_blk_idx)
  # pl.debug_print("[jevin debug] q_blk_idx={}", q_blk_idx)

  def create_kv_async_copy_descriptors(
      heads_blk_idx, seq_idx, kv_blk_idx, buf_idx
  ):
    offset = (seq_idx, kv_blk_idx * num_kv_pages_per_blk)
    heads_start = heads_blk_idx * num_kv_heads_per_blk
    async_copy_k = MultiPageAsyncCopyDescriptor(
        k_pages_hbm_ref.at[:, :, pl.ds(heads_start, num_kv_heads_per_blk), :],
        k_bufs.at[buf_idx],
        sems.at[buf_idx, 0],
        page_indices_ref,
        offset,
    )
    async_copy_v = MultiPageAsyncCopyDescriptor(
        v_pages_hbm_ref.at[:, :, pl.ds(heads_start, num_kv_heads_per_blk), :],
        v_bufs.at[buf_idx],
        sems.at[buf_idx, 1],
        page_indices_ref,
        offset,
    )
    return async_copy_k, async_copy_v

  def strided_load_kv(ref, start, step):
    if ref.dtype == jnp.float32:
      return ref[start::step, :]
    packing = get_dtype_packing(ref.dtype)
    assert ref.dtype == jnp.bfloat16
    assert step % packing == 0
    b_start = start // packing
    b_offset = start % packing
    b_step = step // packing
    b_ref = ref.bitcast(jnp.int32)
    b = b_ref[b_start::b_step, :]
    bw = 32 // packing
    b = jnp.right_shift(b, bw * b_offset)
    b = jnp.left_shift(b, bw * (packing - 1))
    return pltpu.bitcast(b, jnp.float32).astype(jnp.bfloat16)

  @pl.when(heads_blk_idx + q_blk_idx == 0)
  def prefetch_first_kv_blk():
    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        heads_blk_idx, init_seq_idx, 0, init_buf_idx
    )
    async_copy_k.start()
    async_copy_v.start()
    # pl.debug_print("[jevin debug] -- START: prefetch_first_kv_blk")
    # pl.debug_print("[jevin debug] heads_blk_idx={}", heads_blk_idx)
    # pl.debug_print("[jevin debug] init_seq_idx={}", init_seq_idx)
    # pl.debug_print("[jevin debug] kv_blk_idx={}", 0)
    # pl.debug_print("[jevin debug] init_buf_idx={}", init_buf_idx)
    # pl.debug_print("[jevin debug] -- END: prefetch_first_kv_blk")

  def is_cur_q_blk_needed(q_states):
    done, cur_seq_idx, _ = q_states
    return jnp.logical_and(done == 0, cur_seq_idx < num_seqs)

  def compute_with_cur_q_blk(q_states):
    done, cur_seq_idx, cur_buf_idx = q_states
    q_start = cu_q_lens_ref[cur_seq_idx]
    q_end = cu_q_lens_ref[cur_seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[cur_seq_idx]

    def get_next_prefetch_ids(
        heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
    ):
      next_kv_blk_idx = kv_blk_idx + 1
      is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
      next_kv_blk_idx = lax.select(
          is_last_kv_blk,
          0,
          next_kv_blk_idx,
      )
      is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
      next_seq_idx = lax.select(
          is_last_kv_blk,
          lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
          cur_seq_idx,
      )
      is_last_seq = next_seq_idx == num_seqs
      next_seq_idx = lax.select(
          is_last_seq,
          0,
          next_seq_idx,
      )
      next_heads_blk_idx = lax.select(
          is_last_seq,
          heads_blk_idx + 1,
          heads_blk_idx,
      )
      next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
      return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

    def flash_attention(
        q,  # [num_q_per_blk * num_q_heads_per_kv_head, head_dim]
        k,  # [num_kv_per_blk, head_dim]
        v,  # [num_kv_per_blk, head_dim]
        head_l_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_m_ref,  # [num_q_per_blk * num_q_heads_per_kv_head, 128]
        head_o_ref,  # [num_q_per_blk, num_q_heads_per_kv_head, head_dim]
        *,
        kv_blk_idx,
    ):
      assert q.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          head_dim,
      )
      assert k.shape == (num_kv_per_blk, head_dim), f"{k.shape=}, {(num_kv_per_blk, head_dim)=} {k.dtype=}"
      assert v.shape == (num_kv_per_blk, head_dim)
      assert head_m_ref.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          128,
      )
      assert head_l_ref.shape == (
          num_q_per_blk * num_q_heads_per_kv_head,
          128,
      )
      assert head_o_ref.shape == (
          num_q_per_blk,
          num_q_heads_per_kv_head,
          head_dim,
      )
      kv_len_start = kv_blk_idx * num_kv_per_blk

      def masked_store(ref, val, start, end, group=1):
        iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
        mask = jnp.logical_and(iota >= start, iota < end)
        pl.store(ref, tuple(slice(None) for _ in ref.shape), val, mask=mask)

      qk = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32)
      qk = qk * sm_scale

      store_start = jnp.maximum(q_start - q_len_start, 0)
      store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

      @pl.when(kv_blk_idx == 0)
      def init_scratch_ref():
        # TODO(jevinjiang): use scratch for f32 temp output
        masked_store(
            head_m_ref,
            jnp.full_like(head_m_ref, -jnp.inf),
            store_start,
            store_end,
            num_q_heads_per_kv_head,
        )
        masked_store(
            head_l_ref,
            jnp.zeros_like(head_l_ref),
            store_start,
            store_end,
            num_q_heads_per_kv_head,
        )
        masked_store(
            head_o_ref,
            jnp.zeros_like(head_o_ref),
            store_start,
            store_end,
        )

      row_ids = (
          (kv_len - q_len)
          + q_len_start
          - q_start
          + jax.lax.broadcasted_iota(
              jnp.int32,
              (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
              0,
          )
          // num_q_heads_per_kv_head
      )

      col_ids = kv_len_start + jax.lax.broadcasted_iota(
          jnp.int32,
          (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
          1,
      )
      causal_mask = row_ids < col_ids
      qk += jnp.where(causal_mask, mask_value, 0.0)

      m_curr = jnp.max(qk, axis=1, keepdims=True)
      s_curr = jnp.exp(qk - m_curr)
      qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)

      lm_store_shape = head_m_ref.shape
      m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
      l_curr = jnp.broadcast_to(
          s_curr.sum(axis=1, keepdims=True), lm_store_shape
      )
      m_prev = head_m_ref[...]
      l_prev = head_l_ref[...]
      m_next = jnp.maximum(m_prev, m_curr)
      masked_store(
          head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head
      )

      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_alpha = alpha * l_prev
      l_next = l_alpha + beta * l_curr
      l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
      masked_store(
          head_l_ref,
          l_next_safe,
          store_start,
          store_end,
          num_q_heads_per_kv_head,
      )

      def broadcast_to_shape(arr, shape):
        if arr.shape == shape:
          return arr
        assert len(arr.shape) == len(shape)
        assert arr.shape[0] == shape[0]
        assert shape[1] % arr.shape[1] == 0
        # no-op concatenation.
        return jnp.concatenate(
            [arr for _ in range(shape[1] // arr.shape[1])], axis=1
        )

      o_curr = head_o_ref[...].reshape(-1, head_dim)
      l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
      beta = broadcast_to_shape(beta, qkv.shape)
      l_next_safe = broadcast_to_shape(l_next_safe, qkv.shape)
      out = lax.div(
          l_alpha * o_curr + beta * qkv,
          l_next_safe,
      ).astype(head_o_ref.dtype)

      masked_store(
          head_o_ref,
          out.reshape(head_o_ref.shape),
          store_start,
          store_end,
      )

    def is_valid_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, _ = kv_states
      return kv_blk_idx * num_kv_per_blk < kv_len

    def compute_with_kv_blk_in_cur_seq(kv_states):
      kv_blk_idx, cur_buf_idx = kv_states
      next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = (
          get_next_prefetch_ids(
              heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
          )
      )

      # pl.debug_print("[jevin debug] -- START: wait_cur_kv_blk")
      # pl.debug_print("[jevin debug] heads_blk_idx={}", heads_blk_idx)
      # pl.debug_print("[jevin debug] cur_seq_idx={}", cur_seq_idx)
      # pl.debug_print("[jevin debug] kv_blk_idx={}", kv_blk_idx)
      # pl.debug_print("[jevin debug] cur_buf_idx={}", cur_buf_idx)
      # pl.debug_print("[jevin debug] -- END: wait_cur_kv_blk")

      @pl.when(next_heads_blk_idx < num_heads_blks)
      def prefetch_next_kv_blk():
        next_async_copy_k, next_async_copy_v = create_kv_async_copy_descriptors(
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
        )
        # pl.debug_print("[jevin debug] -- START: prefetch_next_kv_blk")
        # pl.debug_print(
        #     "[jevin debug] next_heads_blk_idx={}", next_heads_blk_idx
        # )
        # pl.debug_print("[jevin debug] next_seq_idx={}", next_seq_idx)
        # pl.debug_print("[jevin debug] next_kv_blk_idx={}", next_kv_blk_idx)
        # pl.debug_print("[jevin debug] next_buf_idx={}", next_buf_idx)
        # pl.debug_print("[jevin debug] -- END: prefetch_next_kv_blk")
        next_async_copy_k.start()
        next_async_copy_v.start()

      cur_async_copy_k, cur_async_copy_v = create_kv_async_copy_descriptors(
          heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
      )

      kv_to_load_shape = (
          num_kv_pages_per_blk * page_size * num_kv_heads_per_blk,
          head_dim,
      )
      k_ref = cur_async_copy_k.wait().reshape(kv_to_load_shape)
      v_ref = cur_async_copy_v.wait().reshape(kv_to_load_shape)

      for kv_head_idx in range(num_kv_heads_per_blk):
        q_head_idx = kv_head_idx * num_q_heads_per_kv_head
        # TODO(jevinjiang): extra handlig for packed type!
        q = q_ref[
            :, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :
        ].reshape(-1, head_dim)
        k = strided_load_kv(k_ref, kv_head_idx, num_kv_heads_per_blk)
        v = strided_load_kv(v_ref, kv_head_idx, num_kv_heads_per_blk)
        flash_attention(
            q,
            k,
            v,
            l_ref.at[kv_head_idx],
            m_ref.at[kv_head_idx],
            o_ref.at[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :],
            kv_blk_idx=kv_blk_idx,
        )

      return kv_blk_idx + 1, next_buf_idx

    _, next_buf_idx = lax.while_loop(
        is_valid_kv_blk_in_cur_seq,
        compute_with_kv_blk_in_cur_seq,
        (0, cur_buf_idx),
    )

    next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
    done = lax.select(q_end < q_len_end, done, 1)
    return done, next_seq_idx, next_buf_idx

  _, seq_idx, buf_idx = lax.while_loop(
      is_cur_q_blk_needed,
      compute_with_cur_q_blk,
      (0, init_seq_idx, init_buf_idx),
  )
  seq_buf_idx_ref[0] = seq_idx
  seq_buf_idx_ref[1] = buf_idx


def ceil_div(a, b):
  assert b != 0
  return (a + b - 1) // b


def get_dtype_packing(dtype):
  if dtype == jnp.float32:
    return 1
  if dtype == jnp.bfloat16:
    return 2
  if dtype == jnp.int8:
    return 4
  if dtype == jnp.int4:
    return 8
  raise ValueError(f"Unsupported dtype: {dtype}")


def get_min_heads_per_blk(num_q_heads, num_kv_heads, dtype):
  packing = get_dtype_packing(dtype)

  def can_be_xla_fully_tiled(x):
    if x % packing != 0:
      return False
    x //= packing
    return x in (1, 2, 4, 8) or x % 8 == 0

  # TODO(jevinjiang): support unaligned number of heads!
  if not can_be_xla_fully_tiled(num_q_heads):
    raise ValueError(
        f"Not implemented: {num_q_heads=} can not be XLA fully tiled."
    )
  if not can_be_xla_fully_tiled(num_kv_heads):
    raise ValueError(
        f"Not implemented: {num_kv_heads=} can not be XLA fully tiled."
    )
  assert num_q_heads % num_kv_heads == 0
  ratio = num_q_heads // num_kv_heads
  # TODO(jevinjiang): we can choose smaller tiling for packed type if large
  # second minor tiling is not on.
  max_kv_tiling = 8 * packing
  min_kv_heads = (
      max_kv_tiling if num_kv_heads >= max_kv_tiling else num_kv_heads
  )
  min_q_heads = min_kv_heads * ratio
  return min_q_heads, min_kv_heads


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "mask_value",
        "num_seqs",
        "sm_scale",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [total_num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[num_seqs]
    page_indices: jax.Array,  # i32[num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[num_seqs + 1]
    # TODO(jevinjiang): make num_seqs dynamic! Pass it as array so we can pass
    # it as prefetch for kernel.
    num_seqs,  # int
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    num_kv_pages_per_block: int = 16,
    num_queries_per_block: int = 128,
    sm_scale: float = 1.0,
):
  # check_kernel_input(
  #     q,
  #     k_pages,
  #     v_pages,
  #     kv_lens,
  #     page_indices,
  #     cu_q_lens,
  #     num_seqs,
  #     num_kv_pages_per_block,
  # )
  num_q, num_q_heads, head_dim = q.shape
  _, page_size, num_kv_heads, _ = k_pages.shape
  num_q_per_blk = num_queries_per_block
  num_kv_pages_per_blk = num_kv_pages_per_block
  num_q_heads_per_kv_head = num_q_heads // num_kv_heads
  # pages_per_seq = page_indices.shape[1]
  # TODO(jevinjiang): did we check pages_per_seq % num_kv_pages_per_blk?
  # num_kv_blks = ceil_div(pages_per_seq, num_kv_pages_per_blk)
  num_q_blks = ceil_div(num_q, num_q_per_blk)

  # TODO(jevinjiang): q.dtype == kv.dtype??
  num_q_heads_per_blk, num_kv_heads_per_blk = get_min_heads_per_blk(
      num_q_heads, num_kv_heads, q.dtype
  )
  # print(f"[jevin debug] {num_q_heads_per_blk=}")
  # print(f"[jevin debug] {num_kv_heads_per_blk=}")

  assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
  num_heads_blks = num_q_heads // num_q_heads_per_blk

  grid = (num_heads_blks, num_q_blks)
  # print(f"[jevin debug] {grid=}")

  def q_index_map(heads_blk_idx, q_blk_idx, *_):
    return (q_blk_idx, heads_blk_idx, 0)

  q_block_spec = pl.BlockSpec(
      (num_q_per_blk, num_q_heads_per_blk, head_dim),
      q_index_map,
  )

  in_specs = [
      q_block_spec,
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
      pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
  ]

  out_specs = q_block_spec

  lm_scratch = pltpu.VMEM(
      # TODO(jevinjiang): use 128 instead of 1 is due to Mosaic does not support
      # unaligned slicing!
      (num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128),
      jnp.float32,
  )

  double_buf_scratch = pltpu.VMEM(
      (
          2,  # For double buffering during DMA copies.
          num_kv_pages_per_blk,
          page_size,
          num_kv_heads_per_blk,
          head_dim,
      ),
      k_pages.dtype,
  )

  scratch_shapes = [
      double_buf_scratch,  # k_bufs
      double_buf_scratch,  # v_bufs
      pltpu.SemaphoreType.DMA((2, 2)),
      lm_scratch,  # l_ref
      lm_scratch,  # m_ref
  ]

  scalar_prefetches = (
      kv_lens,
      page_indices,
      cu_q_lens,
      jnp.array((0, 0), jnp.int32),  # seq_idx, buf_idx
  )

  kernel = pl.pallas_call(
      functools.partial(ragged_paged_attention_kernel, mask_value=mask_value, sm_scale=sm_scale),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          in_specs=in_specs,
          out_specs=out_specs,
          grid=grid,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=(
              "arbitrary",
              "arbitrary",
          ),
          vmem_limit_bytes=1024 * 1024 * 36,
      ),
      out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=jnp.float32),
      # interpret=True,
      # debug=True,
      name="ragged_paged_attention_kernel_v2_opt",
  )

  return kernel(*scalar_prefetches, q, k_pages, v_pages).astype(q.dtype)