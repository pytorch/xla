import os

import torch
import torch_xla
import torch.cuda
import unittest
import torch_xla.runtime as xr
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm


class GpuDeviceDetectionTest(unittest.TestCase):

  def setUpClass():
    os.unsetenv(xenv.PJRT_DEVICE)
    os.unsetenv(xenv.GPU_NUM_DEVICES)

  def test_automatically_detects_cuda(self):
    unittest.skipIf(not torch.cuda.is_available(),
                    'Not supported when CUDA is not available.')

    device_type = xr.device_type()
    self.assertEqual(device_type, "CUDA")
    self.assertEqual(os.environ[xenv.GPU_NUM_DEVICES], "1")

    supported_devices = xm.get_xla_supported_devices("CUDA")
    self.assertTrue(len(supported_devices) > 0)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
