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
    self.assertEqual(self.num_cuda_devices, xr.local_device_count())

  def test_num_global_device_count(self):
    self.assertEqual(self.num_cuda_devices, xr.global_device_count())

  def test_local_process_count(self):
    self.assertEqual(1, xr.local_process_count())

  def test_world_size(self):
    # world_size is the number of processes participating in the job per https://pytorch.org/docs/stable/distributed.html.
    self.assertEqual(1, xr.world_size())

  def test_addressable_device_count(self):
    self.assertEqual(self.num_cuda_devices, xr.addressable_device_count())

  def test_addressable_runtime_device_count(self):
    self.assertEqual(self.num_cuda_devices,
                     xr.addressable_runtime_device_count())

  def test_ordinal(self):
    self.assertEqual(0, xr.local_ordinal())
    self.assertEqual(0, xr.global_ordinal())

  def test_process_index(self):
    self.assertEqual(0, xr.process_index())

  def test_process_count(self):
    self.assertEqual(1, xr.process_count())


if __name__ == '__main__':
  absltest.main()
