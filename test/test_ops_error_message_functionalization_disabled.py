import os
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"

import expecttest
import torch
import torch_xla
import unittest


class TestOpsErrorMessageFunctionalizationDisabled(expecttest.TestCase):

  def test_expand_raises_error_on_higher_rank_tensor(self):
    device = torch_xla.device()
    a = torch.rand(1, 1, 2, 3, device=device)
    sizes = [-1, 3]

    def test():
      return a.expand(sizes)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""expand(): expected the `input` tensor f32[1,1,2,3] (rank: 4) to have a rank smaller or equal to the given `sizes` [-1, 3] (rank: 2)."""
    )

  def test_expand_raises_error_on_size_mismatch(self):
    device = torch_xla.device()
    a = torch.rand(1, 1, 2, 3, device=device)
    sizes = [1, 1, 1, 3]

    def test():
      return a.expand(sizes)

    self.assertExpectedRaisesInline(
        exc_type=RuntimeError,
        callable=test,
        expect="""expand(): expected dimension 2 of the given `sizes` [1, 1, 1, 3] (1) to be -1, or equal to the size of the `input` tensor f32[1,1,2,3] at dimension 2 (2)."""
    )


if __name__ == "__main__":
  unittest.main()
