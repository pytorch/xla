import logging
import torch
from torch import nn as nn
import unittest

import torch_xla.experimental.torch_triton as torch_triton
import torch_xla
from torch_xla import runtime as xr

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


class TritonTest(unittest.TestCase):

  @unittest.skipIf(xr.device_type() != 'CUDA', "This test only works on GPU.")
  def test_gpu_custom_call_triton_add(self):
    size = 16

    x = torch.arange(size, dtype=torch.int64).to("xla")
    y = torch.arange(size, dtype=torch.int64).to("xla")
    output = torch.empty_like(x)
    block_size = 8
    grid = (size // block_size,)
    payload = torch_triton.triton_call(
        x, y, output, size, kernel=add_kernel, grid=grid, BLOCK_SIZE=block_size)
    output = torch_xla._XLAC._xla_gpu_custom_call(
      [x,y], payload, [output.shape],
      [torch.int64])
    output_torch = x + y
    self.assertTrue(torch.allclose(output[0].cpu(), output_torch.cpu()))

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
