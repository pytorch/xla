import os
import torch_xla
import unittest
import warnings


def onlyOnCUDA(fn):
  accelerator = os.environ.get("PJRT_DEVICE", "").lower()
  return unittest.skipIf(accelerator != "cuda", "PJRT_DEVICE=CUDA required")(fn)


class CUDAWarningTest(unittest.TestCase):

  @onlyOnCUDA
  def test_raise_warning_on_cuda(self):
    with warnings.catch_warnings(record=True) as w:
      torch_xla.device()
      self.assertEqual(len(w), 1)
      self.assertRegex(
          text=str(w[0].message),
          expected_regex="The XLA:CUDA device is deprecated in release 2.8."
      )


if __name__ == '__main__':
  test = unittest.main()
