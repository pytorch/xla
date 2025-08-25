import os
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import unittest


class TestClientInitializationError(unittest.TestCase):

  def test(self):

    def initialize_client(device):
      os.environ[xenv.PJRT_DEVICE] = device

      # The message does not change!
      # After the first call with DUMMY_DEVICE, all other calls will have
      # "DUMMY_DEVICE" in their message.
      message = (
          f"No PjRtPlugin registered for: DUMMY_DEVICE. "
          f"Make sure the environment variable {xenv.PJRT_DEVICE} is set "
          "to a correct device name.")

      with self.assertRaisesRegex(RuntimeError, expected_regex=message):
        torch_xla._XLAC._init_computation_client()

    # Run the initialization function the first time, ending up in an
    # exception thrown.
    initialize_client("DUMMY_DEVICE")

    # Even if the device exists, this call should fail, since the result
    # of the first call is cached.
    initialize_client("CPU")


if __name__ == '__main__':
  unittest.main()
