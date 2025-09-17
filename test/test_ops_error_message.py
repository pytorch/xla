import expecttest
import os
import torch
import torch_xla
import unittest


def onlyOnCPU(fn):
  accelerator = os.environ.get("PJRT_DEVICE", "").lower()
  return unittest.skipIf(accelerator != "cpu", "PJRT_DEVICE=CPU required")(fn)


class TestOpsErrorMessage(expecttest.TestCase):

  def test_add_broadcast_error(self):
    a = torch.rand(2, 2, 4, 4, device="xla")
    b = torch.rand(2, 2, device="xla")

    def test():
      return torch.add(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""Shapes are not compatible for broadcasting: f32[2,2,4,4] vs. f32[2,2]. Expected dimension 2 of shape f32[2,2,4,4] (4) to match dimension 0 of shape f32[2,2] (2). Either that or that any of them is either 1 or unbounded. Try reshaping one of the tensors to match the other."""
    )

  @onlyOnCPU
  def test_construct_large_tensor_raises_error(self):

    def test():
      # When eager-mode is enabled, OOM is triggered here.
      a = torch.rand(1024, 1024, 1024, 1024, 1024, device=torch_xla.device())
      b = a.sum()
      # OOM is raised when we try to bring data from the device.
      return b.cpu()

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""Error preparing computation: Out of memory allocating 4503599761588224 bytes."""
    )

  def test_cat_raises_error_on_incompatible_shapes(self):
    a = torch.rand(2, 2, device=torch_xla.device())
    b = torch.rand(5, 1, device=torch_xla.device())

    def test():
      return torch.cat([a, b])

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""cat(): cannot concatenate tensors of shape f32[2,2] with f32[5,1] at dimension 0. Expected shapes to be equal (except at dimension 0) or that either of them was a 1D empty tensor of size (0,)."""
    )

  def test_div_raises_error_on_invalid_rounding_mode(self):
    a = torch.rand(2, 2, device=torch_xla.device())

    def test():
      return torch.div(a, 2, rounding_mode="bad")

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""div(): invalid rounding mode `bad`. Expected it to be either 'trunc', 'floor', or be left unspecified."""
    )

  def test_flip_raises_error_on_duplicated_dims(self):
    a = torch.rand(2, 2, 2, 2, device=torch_xla.device())
    dims = [0, 0, 0, 1, 2, 3, -1]

    def test():
      return torch.flip(a, dims=dims)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""flip(): expected each dimension to appear at most once. Found dimensions: 0 (3 times), 3 (2 times). Consider changing dims from [0, 0, 0, 1, 2, 3, -1] to [0, 1, 2, 3]."""
    )

  def test_full_raises_error_on_negative_size(self):
    shape = [2, -2, 2]

    def test():
      return torch.full(shape, 1.5, device="xla")

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""full(): expected concrete sizes (i.e. non-symbolic) to be positive values. However found negative ones: [2, -2, 2]."""
    )

  def test_gather_raises_error_on_rank_mismatch(self):
    S = 2

    input = torch.arange(4, device=torch_xla.device()).view(S, S)
    index = torch.randint(0, S, (S, S, S), device=torch_xla.device())
    dim = 1

    def test():
      return torch.gather(input, dim, index)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""gather(): expected rank of input (2) and index (3) tensors to be the same."""
    )

  def test_gather_raises_error_on_invalid_index_size(self):
    S = 2
    X = S + 2

    input = torch.arange(16, device=torch_xla.device()).view(S, S, S, S)
    index = torch.randint(0, S, (X, S, X, S), device=torch_xla.device())
    dim = 1

    def test():
      return torch.gather(input, dim, index)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""gather(): expected sizes of index [4, 2, 4, 2] to be smaller or equal those of input [2, 2, 2, 2] on all dimensions, except on dimension 1. However, that's not true on dimensions [0, 2]."""
    )

  def test_random__raises_error_on_empty_interval(self):
    a = torch.empty(10, device=torch_xla.device())
    from_ = 3
    to_ = 1

    def test():
      return a.random_(from_, to_)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""random_(): expected `from` (3) to be smaller than `to` (1)."""
    )

  def test_random__raises_error_on_value_out_of_type_value_range(self):
    a = torch.empty(10, device=torch_xla.device(), dtype=torch.float16)
    from_ = 3
    to_ = 65_504 + 2

    def test():
      return a.random_(from_, to_)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""random_(): expected `to` to be within the range [-65504, 65504]. However got value 65505, which is greater than the upper bound."""
    )

  def test_mm_raises_error_on_non_matrix_input(self):
    device = torch_xla.device()
    a = torch.rand(2, 2, 2, device=device)
    b = torch.rand(2, 2, device=device)

    def test():
      return torch.mm(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""mm(): expected the first input tensor f32[2,2,2] to be a matrix (i.e. a 2D tensor)."""
    )

  def test_mm_raises_error_on_incompatible_shapes(self):
    device = torch_xla.device()
    a = torch.rand(2, 5, device=device)
    b = torch.rand(8, 2, device=device)

    def test():
      return torch.mm(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""mm(): cannot matrix-multiply tensors f32[2,5] and f32[8,2]. Expected the size of dimension 1 of the first input tensor (5) to be equal the size of dimension 0 of the second input tensor (8)."""
    )

  def test_roll_raises_error_on_empty_shifts(self):
    device = torch_xla.device()
    a = torch.rand(2, 2, 2, device=device)
    shifts = []

    def test():
      return torch.roll(a, shifts)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""roll(): expected `shifts` to have at least 1 element.""")

  def test_roll_raises_error_on_shifts_with_empty_dims(self):
    device = torch_xla.device()
    a = torch.rand(2, 2, 2, device=device)
    shifts = [2, 2]

    def test():
      return torch.roll(a, shifts)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""roll(): expected `shifts` [2, 2] (size=2) to have exactly 1 element when `dims` is empty."""
    )

  def test_roll_raises_error_on_mismatched_dims_and_shifts(self):
    device = torch_xla.device()
    a = torch.rand(2, 2, 2, device=device)
    shifts = [2, 2]
    dims = [0]

    def test():
      return torch.roll(a, shifts, dims)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""roll(): expected `dims` [0] (size=1) to match the size of `shifts` [2, 2] (size=2)."""
    )

  def test_trace_raises_error_on_non_matrix_input(self):
    device = torch_xla.device()
    a = torch.rand(2, 2, 2, device=device)

    def test():
      torch.trace(a)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""trace(): expected the input tensor f32[2,2,2] to be a matrix (i.e. a 2D tensor)."""
    )

  def test_clamp_scalar_raises_error_on_no_min_and_max(self):
    device = torch_xla.device()
    a = torch.rand(2, 5, device=device)

    def test():
      # Dispatch to `clamp()` overload explicitly.
      # Otherwise, it's dispatched to `clamp.Tensor()`, which doesn't have
      # this check.
      return torch.ops.aten.clamp.default(a)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""clamp(): expected at least one of `min` or `max` arguments to be specified."""
    )

  def test_bmm_raises_error_on_non_3D_tensor_input(self):
    device = torch_xla.device()
    a = torch.rand(2, 3, 4, device=device)
    b = torch.rand(2, 4, 3, device=device)

    def test_a():
      torch.bmm(a[0], b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test_a,
        expect="""bmm(): expected `input` f32[3,4] (a 2D tensor), the 1st input tensor, to be a 3D tensor."""
    )

    def test_b():
      torch.bmm(a, b[0])

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test_b,
        expect="""bmm(): expected `mat2` f32[4,3] (a 2D tensor), the 2nd input tensor, to be a 3D tensor."""
    )

  def test_bmm_raises_error_on_different_batch_dimension(self):
    device = torch_xla.device()
    a = torch.rand(4, 3, 4, device=device)
    b = torch.rand(2, 4, 3, device=device)

    def test():
      torch.bmm(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""bmm(): expected the size of the batch dimension (i.e. dimension 0) of `input` f32[4,3,4] (batch dimension size: 4), the 1st input tensor, to be the same as the size of the batch dimension of `mat2` f32[2,4,3] (batch dimension size: 2), the 2nd input tensor."""
    )

  def test_bmm_raises_error_on_incompatible_shapes(self):
    device = torch_xla.device()
    a = torch.rand(2, 3, 8, device=device)
    b = torch.rand(2, 4, 3, device=device)

    def test():
      torch.bmm(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""bmm(): cannot apply batch matrix-multiplication to `input` f32[2,3,8], the 1st input tensor, and to `mat2` f32[2,4,3], the 2nd input tensor. Expected the size of dimension 2 of `input` (8) to be equal the size of dimension 1 of `mat2` (4)."""
    )

  def test_baddbmm_raises_error_on_incompatible_shapes(self):
    device = torch_xla.device()
    input = torch.rand(3, 3, device=device)
    a = torch.rand(2, 3, 8, device=device)
    b = torch.rand(2, 4, 3, device=device)

    def test():
      torch.baddbmm(input, a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""baddbmm(): cannot apply batch matrix-multiplication to `batch1` f32[2,3,8], the 2nd input tensor, and to `batch2` f32[2,4,3], the 3rd input tensor. Expected the size of dimension 2 of `batch1` (8) to be equal the size of dimension 1 of `batch2` (4)."""
    )

  def test_uniform__raises_error_on_invalid_range(self):
    device = torch_xla.device()
    a = torch.empty(5, 5, device=device)
    from_ = 5.
    to_ = 2.

    def test():
      return a.uniform_(from_, to_)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""uniform_(): expected `from` (5) <= `to` (2).""")

  def test_avg_pool_3d_raises_error_on_bad_spec(self):
    device = torch_xla.device()
    a = torch.rand(1, 1, 4, 4, 4, device=device)

    def gen_test_fn(kernel_size=[2, 2, 2], stride=[], padding=[0]):
      return lambda: torch.nn.functional.avg_pool3d(a, kernel_size, stride, padding)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=gen_test_fn(kernel_size=[2, 2]),
        expect="""avg_pool3d(): expected argument kernel_size [2, 2] (size: 2) to have size of 3."""
    )

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=gen_test_fn(stride=[1, 2]),
        expect="""avg_pool3d(): expected argument stride [1, 2] (size: 2) to have size of 3."""
    )

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=gen_test_fn(padding=[1, 2]),
        expect="""avg_pool3d(): expected argument padding [1, 2] (size: 2) to have size of 3."""
    )


if __name__ == "__main__":
  unittest.main()
