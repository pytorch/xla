import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import unittest
import test_utils
import pytorch_test_base

from torch.testing import make_tensor
from itertools import product
from functools import partial
from torch.testing._internal.common_utils import TestCase, run_tests, IS_JETSON
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, dtypes)
from torch.testing._internal.common_dtype import (all_types_and_complex_and)


# These tests are a copy of upstream pytorch tests due to the way lazy tensors
# work. The randperm op generates a random tensor. Every iteration of the test
# recompiles the randperm op thus generating a different random tensor which
# makes the test non-deterministic. To force determinism, this test has to
# call PyTorch/XLA `torch_xla.sync()` to materialize the tensor rather than
# recompile.
class TestPythonOps(pytorch_test_base.XLATestBase):

  @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
  def test_put(self, dtype):
    if dtype in self.unsupported_dtypes:
      raise unittest.SkipTest("Dtype {0} is unsupported by XLA".format(
          str(dtype)))

    device = xm.xla_device()
    real_device_type = xm.xla_device_hw(str(xm.xla_device()))
    if real_device_type == "TPU":
      raise unittest.SkipTest("TestPut is too slow on TPU. Skipped")

    src_size = (4,)

    make_arg = partial(make_tensor, device=device, dtype=dtype)
    make_idx = partial(make_tensor, low=0, device=device, dtype=torch.int64)

    def ref_put(dst, idx, src, accumulate):
      new_dst = dst.clone(memory_format=torch.contiguous_format).view(-1)
      new_idx = idx.contiguous().view(-1)
      new_src = src.contiguous().view(-1)
      method = new_dst.index_add_ if accumulate else new_dst.index_copy_
      return method(0, new_idx, new_src).view_as(dst)

    for dst_contig, src_contig, idx_contig, idx_reshape, accumulate in product(
        [True, False], repeat=5):
      for dst_size in ((5,), (4, 5)):
        dst = make_arg(dst_size, noncontiguous=not dst_contig)
        src = make_arg(src_size, noncontiguous=not src_contig)

        # If accumulate=True, `put_` should be deterministic regardless of the inputs on CPU
        # On CUDA it may not be, but the test has enough tolerance to account for this
        if accumulate:
          idx = make_idx(src_size, high=dst.numel())
        else:
          idx = torch.randperm(
              dst.numel(), dtype=torch.int64, device=device)[:src_size[0]]
        if not idx_contig:
          idx = torch.repeat_interleave(idx, 2, dim=-1)[..., ::2]
        if idx_reshape:
          idx = idx.reshape(2, 2)
        out = torch.put(dst, idx, src, accumulate)

        torch_xla.sync()

        # out-place
        reference = ref_put(dst, idx, src, accumulate)
        self.assertEqual(out, reference)

        # in-place
        dst.put_(idx, src, accumulate)
        self.assertEqual(dst, reference)

    # Create the 8 possible combinations of scalar sizes for target / index / source
    scalars = ((make_arg(size_t), make_idx(size_i, high=1), make_arg(size_s))
               for size_t, size_i, size_s in product([(), (1,)], repeat=3))
    for (dest, idx, source), accumulate in product(scalars, [True, False]):
      dest_init = dest.clone()
      # out-place
      out = torch.put(dest, idx, source, accumulate=accumulate)
      # in-place
      dest1 = dest.clone()
      dest1.put_(idx, source, accumulate=accumulate)
      for d in [out, dest1]:
        if accumulate:
          self.assertEqual(d.item(), (dest_init + source).item())
        else:
          self.assertEqual(d.item(), source.item())

    # Empty case
    dest = make_arg((3, 2))
    reference = dest.clone()
    idx = make_idx((0,), high=1)
    source = make_arg((0,))
    for accumulate in [True, False]:
      out = torch.put(dest, idx, source, accumulate=accumulate)
      self.assertEqual(out, reference)
      dest.put_(idx, source, accumulate=accumulate)
      self.assertEqual(dest, reference)

  @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
  def test_index_copy(self, dtype):
    if dtype in self.unsupported_dtypes:
      raise unittest.SkipTest("Dtype {0} is unsupported by XLA".format(
          str(dtype)))

    device = xm.xla_device()

    # We just test for num_copy <= num_dest, as otherwise there are repeated indices
    # and the behavior is undefined
    num_copy, num_dest = 3, 5

    def make_arg(batch_sizes, n, dim, contig):
      size_arg = batch_sizes[:dim] + (n,) + batch_sizes[dim:]
      return make_tensor(
          size_arg,
          dtype=dtype,
          device=device,
          low=None,
          high=None,
          noncontiguous=not contig)

    def ref_index_copy(tgt, dim, idx, src):
      for i in range(idx.size(0)):
        idx_dest = dim * (slice(None),) + (idx[i],)
        idx_src = dim * (slice(None),) + (i,)
        tgt[idx_dest] = src[idx_src]

    # More thorough testing as in index_add
    for dest_contig, src_contig, index_contig in product([True, False],
                                                         repeat=3):
      for other_sizes in ((), (4, 5)):
        for dim in range(len(other_sizes)):
          dest = make_arg(other_sizes, num_dest, dim, dest_contig)
          src = make_arg(other_sizes, num_copy, dim, src_contig)
          idx = torch.randperm(
              num_dest, dtype=torch.int64, device=device)[:num_copy]
          if not index_contig:
            idx = torch.repeat_interleave(idx, 2, dim=-1)
            idx = idx[..., ::2]

          torch_xla.sync()

          dest2 = dest.clone()
          dest.index_copy_(dim, idx, src)
          ref_index_copy(dest2, dim, idx, src)
          self.assertEqual(dest, dest2)


instantiate_device_type_tests(TestPythonOps, globals())

if __name__ == '__main__':
  run_tests()
