import logging
import torch
from torch import nn as nn
import unittest

import torch_xla.experimental.triton as xla_triton
import torch_xla
from torch_xla import runtime as xr
from torch_xla.test.test_utils import skipIfCUDA

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
  # Triton add kernel from https://github.com/openai/triton/blob/main/python/tutorials/01-vector-add.py#L28
  # There are multiple 'programs' processing different data. We identify which program
  # we are here:
  pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
  # This program will process inputs that are offset from the initial data.
  # For instance, if you had a vector of length 256 and block_size of 64, the programs
  # would each access the elements [0:64, 64:128, 128:192, 192:256].
  # Note that offsets is a list of pointers:
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # Create a mask to guard memory operations against out-of-bounds accesses.
  mask = offsets < n_elements
  # Load x and y from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size.
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  # Write x + y back to DRAM.
  tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr):
  # range of values handled by this stage
  if STAGE == 1:
    lo, hi = 0, start_m * BLOCK_M
  elif STAGE == 2:
    lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    lo = tl.multiple_of(lo, BLOCK_M)
  # causal = False
  else:
    lo, hi = 0, N_CTX
  K_block_ptr = tl.advance(K_block_ptr, (0, lo))
  V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
  # loop over k, v and update accumulator
  for start_n in range(lo, hi, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr)
    qk = tl.dot(q, k)
    if STAGE == 2:
      mask = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
    else:
      m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
      qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    v = tl.load(V_block_ptr)
    if fp8_v:
      p = p.to(tl.float8e5)
    else:
      p = p.to(tl.float16)
    acc = tl.dot(p, v, acc)
    # update m_i and l_i
    m_i = m_ij
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
  return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    STAGE: tl.constexpr  #
):
  tl.static_assert(BLOCK_N <= HEAD_DIM)
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // H
  off_h = off_hz % H
  qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

  # block pointers
  Q_block_ptr = tl.make_block_ptr(
      base=Q + qvk_offset,
      shape=(N_CTX, HEAD_DIM),
      strides=(stride_qm, stride_qk),
      offsets=(start_m * BLOCK_M, 0),
      block_shape=(BLOCK_M, HEAD_DIM),
      order=(1, 0),
  )
  V_block_ptr = tl.make_block_ptr(
      base=V + qvk_offset,
      shape=(N_CTX, HEAD_DIM),
      strides=(stride_vk, stride_vn),
      offsets=(0, 0),
      block_shape=(BLOCK_N, HEAD_DIM),
      order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
      base=K + qvk_offset,
      shape=(HEAD_DIM, N_CTX),
      strides=(stride_kk, stride_kn),
      offsets=(0, 0),
      block_shape=(HEAD_DIM, BLOCK_N),
      order=(0, 1),
  )
  O_block_ptr = tl.make_block_ptr(
      base=Out + qvk_offset,
      shape=(N_CTX, HEAD_DIM),
      strides=(stride_om, stride_on),
      offsets=(start_m * BLOCK_M, 0),
      block_shape=(BLOCK_M, HEAD_DIM),
      order=(1, 0),
  )
  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  # initialize pointer to m and l
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
  acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
  # load scales
  qk_scale = sm_scale
  qk_scale *= 1.44269504  # 1/log(2)
  # load q: it will stay in SRAM throughout
  q = tl.load(Q_block_ptr)
  # stage 1: off-band
  # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
  # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
  if STAGE & 1:
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        4 - STAGE,
        offs_m,
        offs_n,
        N_CTX,
        V.dtype.element_ty == tl.float8e5  #
    )
  # stage 2: on-band
  if STAGE & 2:
    # barrier makes it easier for compielr to schedule the
    # two loops independently
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        2,
        offs_m,
        offs_n,
        N_CTX,
        V.dtype.element_ty == tl.float8e5  #
    )
  # epilogue
  m_i += tl.math.log2(l_i)
  acc = acc / l_i[:, None]
  m_ptrs = M + off_hz * N_CTX + offs_m
  tl.store(m_ptrs, m_i)
  tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class TritonTest(unittest.TestCase):

  @unittest.skipIf(xr.device_type() != 'CUDA', "This test only works on GPU.")
  def test_gpu_custom_call_triton_add(self):
    size = 16

    x = torch.arange(size, dtype=torch.int64).to("xla")
    y = torch.arange(size, dtype=torch.int64).to("xla")
    output = torch.empty_like(x)
    block_size = 8
    grid = (triton.cdiv(size, block_size),)
    payload = xla_triton.triton_call(
        x, y, output, size, kernel=add_kernel, grid=grid, BLOCK_SIZE=block_size)
    output = torch_xla._XLAC._xla_gpu_custom_call([x, y], payload,
                                                  [output.shape], [torch.int64])
    output_torch = x + y
    self.assertTrue(torch.allclose(output[0].cpu(), output_torch.cpu()))

  @unittest.skipIf(xr.device_type() != 'CUDA', "This test only works on GPU.")
  def test_gpu_custom_call_triton_flash_attention(self):
    torch.manual_seed(20)
    Z, H, N_CTX, HEAD_DIM = (1, 2, 1024, 64)
    causal = False
    stage = 3 if causal else 1
    dtype = torch.float16
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="xla")
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="xla")
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="xla")
    sm_scale = 0.5
    # reference implementation
    triangle = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
      p[:, :, triangle == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()

    ref_out = torch.matmul(p, v)
    # triton implementation

    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                    device=q.device,
                    dtype=torch.float32)
    BLOCK_N = 32
    BLOCK_M = 64
    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] *
                         q.shape[1], 1)
    payload = xla_triton.triton_call(
        q,
        k,
        v,
        sm_scale,
        M,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),  #
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),  #
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),  #
        q.shape[0],
        q.shape[1],
        q.shape[2],
        kernel=_attn_fwd,
        grid=grid,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        STAGE=stage)

    output = torch_xla._XLAC._xla_gpu_custom_call([q, k, v, M], payload,
                                                  [o.shape], [torch.float16])
    # compare
    assert torch.allclose(ref_out, output[0], atol=1e-2, rtol=0)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
