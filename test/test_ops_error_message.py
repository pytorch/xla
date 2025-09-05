import expecttest
import os
import torch
import torch_xla
import unittest


def onlyOnCPU(fn):
  accelerator = os.environ.get("PJRT_DEVICE").lower()
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
        expect="""random_(): expected `to` to be within the range [-65504, 65504]. However got value 65505, which is greater than the upper bound.""")

  def test_mm_raises_error_on_non_matrix_input(self):
    device = torch_xla.device()
    a = torch.rand(2, 2, 2, device=device)
    b = torch.rand(2, 2, device=device)

    def test():
      torch.mm(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""random_(): expected `to` to be within the range [-65504, 65504]. However got value 65505, which is greater than the upper bound.""")

  def test_mm_raises_error_on_incompatible_shapes(self):
    device = torch_xla.device()
    a = torch.rand(2, 5, device=device)
    b = torch.rand(8, 2, device=device)

    def test():
      torch.mm(a, b)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""random_(): expected `to` to be within the range [-65504, 65504]. However got value 65505, which is greater than the upper bound.""")
