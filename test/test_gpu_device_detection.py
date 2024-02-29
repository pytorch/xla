import os
import unittest

import torch
import torch.cuda
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


@unittest.skipIf(
    os.getenv(xenv.PJRT_DEVICE) != None or not torch.cuda.is_available(),
    f"SKipping test since PJRT_DEVICE was explicitly set or CUDA is not available.",
)
class GpuDeviceDetectionTest(unittest.TestCase):

  def setUpClass():
    os.unsetenv(xenv.PJRT_DEVICE)
    os.unsetenv(xenv.GPU_NUM_DEVICES)

  def test_automatically_detects_cuda(self):
    device_type = xr.device_type()
    self.assertEqual(device_type, "CUDA")
    self.assertEqual(os.environ[xenv.GPU_NUM_DEVICES],
                     str(torch.cuda.device_count()))

    supported_devices = xm.get_xla_supported_devices("CUDA")
    self.assertTrue(len(supported_devices) > 0)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
