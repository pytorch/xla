import os

import torch
import torch_xla
from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv

class TestExperimentalPjrtMultiCpu(parameterized.TestCase):
    
    def setUp(self):
        pjrt.set_device_type('CPU')

        os.environ.pop(xenv.CPU_ASYNC_CLIENT, None)
        os.environ.pop(xenv.CPU_NUM_DEVICES, None)

    def test_default_cpu_device(self):
        devices_per_process = pjrt.run_multiprocess(xm.xla_device)
        print(devices_per_process)

    def test_multi_cpu_devices(self):
        os.environ.update({
            xenv.CPU_ASYNC_CLIENT: True,
            xenv.CPU_NUM_DEVICES: 4,
        })
        devices_per_process = pjrt.run_multiprocess(xm.xla_device)
        print(devices_per_process)



if __name__ == '__main__':
  absltest.main()