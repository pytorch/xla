import concurrent.futures
import itertools
import os
import queue
import requests
import unittest
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla import runtime as xr
from torch_xla._internal import pjrt
from absl.testing import absltest, parameterized


@unittest.skipIf(xr.device_type() != "CUDA",
                 f"GPU tests should only run on GPU devices.")
class TestExperimentalSingleProcPjrtGpu(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    command = 'nvidia-smi --list-gpus | wc -l'
    result = subprocess.run(
        command,
        capture_output=True,
        shell=True,
        check=True,
        text=True,
    )
    cls.num_cuda_devices = int(result.stdout)

  def test_num_local_devices(self):
    self.assertLen(xm.get_xla_supported_devices(),
                   xr.addressable_device_count())
    self.assertEqual(self.num_cuda_devices, xr.addressable_device_count())

  def test_num_global_devices(self):
    self.assertLen(torch_xla._XLAC._xla_get_all_devices(),
                   xr.global_device_count())
    self.assertEqual(self.num_cuda_devices, xr.global_device_count())


if __name__ == '__main__':
  absltest.main()
