import glob
import os
import textwrap

from absl.testing import absltest, parameterized
import torch_xla.core.xla_env_vars as xenv
from torch_xla.experimental import tpu

from unittest import mock


class TestExperimentalTpu(parameterized.TestCase):

  @parameterized.named_parameters(
      ('default_one_host', None, None),
      ('one_process_one_host', '1,1,1', 1),
      ('multi_process_one_host', '2,2,1', 4),
      ('multi_process_v4-16', '2,2,2', 8),
      ('multi_process_v4-32', '2,2,4', 16),
  )
  def test_process_bounds_size(self, process_bounds, expected):
    envs = {xenv.TPU_PROCESS_BOUNDS: process_bounds} if process_bounds else {}
    with mock.patch.dict(os.environ, envs, clear=True):
      n = tpu.process_bounds_size()

    self.assertEqual(n, expected)

  @parameterized.named_parameters(
      ('no_chips', 0),
      ('one_chip', 1),
      ('four_chips', 4),
  )
  def test_num_available_chips(self, num_tpu_chips):
    vendor_id_files = []
    vendor_ids = ['0x1234', '0x4321', '0xabcd'
                 ] + [tpu._GOOGLE_PCI_VENDOR_ID] * num_tpu_chips
    for vendor in vendor_ids:
      tmpdir = self.create_tempdir()
      vendor_file = tmpdir.create_file('vendor', content=vendor)

      device = tpu._TPU_PCI_DEVICE_IDS[
          0] if vendor == tpu._GOOGLE_PCI_VENDOR_ID else '0x7890'
      tmpdir.create_file('device', content=device)

      vendor_id_files.append(vendor_file.full_path)

    with mock.patch.object(glob, 'glob', return_value=vendor_id_files):
      self.assertEqual(tpu.num_available_chips(), num_tpu_chips)

  @parameterized.named_parameters(
      ('default_one_host', None, 4),
      ('one_process_one_host', '1,1,1', 1),
      ('multi_process_one_host', '2,2,1', 4),
      ('multi_process_v4-16', '2,2,2', 4),
      ('multi_process_v4-32', '2,2,4', 4),
      ('single_chip_default', None, 1, 1),
  )
  def test_num_local_processes(self, process_bounds, expected, num_chips=4):
    envs = {xenv.TPU_PROCESS_BOUNDS: process_bounds} if process_bounds else {}
    with mock.patch.dict(
        os.environ, envs, clear=True), mock.patch.object(
            tpu, 'num_available_chips', return_value=num_chips):
      n = tpu.num_local_processes()

    self.assertEqual(n, expected)

  @parameterized.parameters((None, None), ('0', 0), ('1', 1), ('15', 15))
  def test_task_id(self, task_id, expected):
    envs = {xenv.CLOUD_TPU_TASK_ID: task_id} if task_id else {}
    with mock.patch.dict(os.environ, envs, clear=True):
      i = tpu.task_id()

    self.assertEqual(i, expected)

  @parameterized.named_parameters(
      ('v4',
       textwrap.dedent("""
        ACCELERATOR_TYPE: 'v4-16'
        CHIPS_PER_HOST_BOUNDS: '2,2,1'
        HOST_BOUNDS: '1,1,2'
        TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1'
        TPU_PROCESS_BOUNDS: '1,1,2'
        ZONE: 'us-central2-b'
        WORKER_ID: '0'
      """), {
           'ACCELERATOR_TYPE': 'v4-16',
           'CHIPS_PER_HOST_BOUNDS': '2,2,1',
           'HOST_BOUNDS': '1,1,2',
           'TPU_CHIPS_PER_PROCESS_BOUNDS': '2,2,1',
           'TPU_PROCESS_BOUNDS': '1,1,2',
           'ZONE': 'us-central2-b',
           'WORKER_ID': '0'
       }, 4),
      ('v5',
       textwrap.dedent("""
        ACCELERATOR_TYPE: 'v5abcdefg-16'
        CHIPS_PER_HOST_BOUNDS: '2,2,1'
        HOST_BOUNDS: '1,1,2'
        TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1'
        TPU_PROCESS_BOUNDS: '1,1,2'
        ZONE: 'us-central2-b'
        WORKER_ID: '0'
      """), {
           'ACCELERATOR_TYPE': 'v5abcdefg-16',
           'CHIPS_PER_HOST_BOUNDS': '2,2,1',
           'HOST_BOUNDS': '1,1,2',
           'TPU_CHIPS_PER_PROCESS_BOUNDS': '2,2,1',
           'TPU_PROCESS_BOUNDS': '1,1,2',
           'ZONE': 'us-central2-b',
           'WORKER_ID': '0'
       }, 5),
  )
  def test_tpu_env_from_gce_metadata(self, tpu_env_yaml, expected_env,
                                     expected_version):
    with mock.patch.object(tpu, '_get_metadata', return_value=tpu_env_yaml):
      tpu_env = tpu.get_tpu_env()
      version = tpu.version()
    self.assertDictEqual(tpu_env, expected_env)
    self.assertEqual(version, expected_version)

  @parameterized.named_parameters(
      ('all-vars-set', {
          xenv.TPU_SKIP_MDS_QUERY: '1',
          xenv.TPU_ACCELERATOR_TYPE: 'v4-16',
          xenv.TPU_PROCESS_BOUNDS: '1,2,2',
          xenv.TPU_HOST_BOUNDS: '1,1,2',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1',
          xenv.TPU_CHIPS_PER_HOST_BOUNDS: '2,1,1',
          xenv.CLOUD_TPU_TASK_ID: '1',
          xenv.TPU_WORKER_ID: '0'
      }, {
          xenv.ACCELERATOR_TYPE: 'v4-16',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1',
          xenv.TPU_PROCESS_BOUNDS: '1,2,2',
          xenv.WORKER_ID: '1'
      }),
      ('defaults-only', {
          xenv.TPU_SKIP_MDS_QUERY: '1',
          xenv.TPU_ACCELERATOR_TYPE: 'v4-16',
          xenv.TPU_HOST_BOUNDS: '1,1,2',
          xenv.TPU_CHIPS_PER_HOST_BOUNDS: '2,1,1',
          xenv.TPU_WORKER_ID: '0'
      }, {
          xenv.ACCELERATOR_TYPE: 'v4-16',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '2,1,1',
          xenv.TPU_PROCESS_BOUNDS: '1,1,2',
          xenv.WORKER_ID: '0'
      }),
  )
  def test_tpu_env_from_env_vars(self, envs, expected):
    with mock.patch.dict(os.environ, envs, clear=True):
      tpu_env = tpu.get_tpu_env()
    self.assertDictEqual(tpu_env, expected)

  @parameterized.named_parameters(
      ('one_host', 't1v-n-ea9d3291-w-0:12345:10.130.0.31', ['localhost']),
      (
          'four_hosts',
          't1v-n-0f996b37-w-0:12345:10.130.0.26,t1v-n-0f996b37-w-1:12346:10.130.0.27,t1v-n-0f996b37-w-2:12347:10.130.0.25,t1v-n-0f996b37-w-3:12348:10.130.0.28',
          ['10.130.0.26', '10.130.0.27', '10.130.0.25', '10.130.0.28'],
      ),
  )
  def test_get_worker_ips(self, worker_network_endpoints, expected):
    with mock.patch.object(
        tpu, '_get_metadata', return_value=worker_network_endpoints):
      worker_ips = tpu.get_worker_ips()

    self.assertListEqual(worker_ips, expected)

  @parameterized.named_parameters(
      ('v5-4_process_0', {
          'ACCELERATOR_TYPE': 'v5-4',
          xenv.TPU_PROCESS_BOUNDS: '2,2,1',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '1,1,1',
          'WORKER_ID': '0'
      }, ['localhost'], 0, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,1',
          xenv.CLOUD_TPU_TASK_ID:
              '0',
          xenv.TPU_PROCESS_PORT:
              '8476',
          xenv.TPU_PROCESS_ADDRESSES:
              'localhost:8476,localhost:8477,localhost:8478,localhost:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '0',
      }),
      ('v5abcdefg-4_process_0', {
          'ACCELERATOR_TYPE': 'v5abcdefg-4',
          xenv.TPU_PROCESS_BOUNDS: '2,2,1',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '1,1,1',
          'WORKER_ID': '0'
      }, ['localhost'], 0, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,1',
          xenv.CLOUD_TPU_TASK_ID:
              '0',
          xenv.TPU_PROCESS_PORT:
              '8476',
          xenv.TPU_PROCESS_ADDRESSES:
              'localhost:8476,localhost:8477,localhost:8478,localhost:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '0',
      }),
      ('v5abcdefg-16_process_0', {
          'ACCELERATOR_TYPE': 'v5abcdefg-16',
          xenv.TPU_PROCESS_BOUNDS: '2,2,1',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '1,1,1',
          'WORKER_ID': '0'
      }, ['localhost'], 0, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,1',
          xenv.CLOUD_TPU_TASK_ID:
              '0',
          xenv.TPU_PROCESS_PORT:
              '8476',
          xenv.TPU_PROCESS_ADDRESSES:
              'localhost:8476,localhost:8477,localhost:8478,localhost:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '0',
      }),
      ('v4-8_process_0', {
          'ACCELERATOR_TYPE': 'v4-8',
          xenv.TPU_PROCESS_BOUNDS: '1,1,1',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1',
          'WORKER_ID': '0'
      }, ['localhost'], 0, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,1',
          xenv.CLOUD_TPU_TASK_ID:
              '0',
          xenv.TPU_PROCESS_PORT:
              '8476',
          xenv.TPU_PROCESS_ADDRESSES:
              'localhost:8476,localhost:8477,localhost:8478,localhost:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '0',
      }),
      ('v4-8_process_3', {
          'ACCELERATOR_TYPE': 'v4-8',
          xenv.TPU_PROCESS_BOUNDS: '1,1,1',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1',
          'WORKER_ID': '0'
      }, ['localhost'], 3, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,1',
          xenv.CLOUD_TPU_TASK_ID:
              '3',
          xenv.TPU_PROCESS_PORT:
              '8479',
          xenv.TPU_PROCESS_ADDRESSES:
              'localhost:8476,localhost:8477,localhost:8478,localhost:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '3',
      }),
      ('v4-16_worker_1_process_0', {
          'ACCELERATOR_TYPE': 'v4-16',
          xenv.TPU_PROCESS_BOUNDS: '1,1,2',
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1',
          'WORKER_ID': '1'
      }, ['10.130.0.31', '10.130.0.30'], 0, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,2',
          xenv.CLOUD_TPU_TASK_ID:
              '4',
          xenv.TPU_PROCESS_PORT:
              '8476',
          xenv.TPU_PROCESS_ADDRESSES:
              '10.130.0.31:8476,10.130.0.31:8477,10.130.0.31:8478,10.130.0.31:8479,10.130.0.30:8476,10.130.0.30:8477,10.130.0.30:8478,10.130.0.30:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '0',
      }),
      # TODO: remove this case when process bounds are added to metadata
      ('v3-8_process_0', {
          'ACCELERATOR_TYPE': 'v3-8',
          'WORKER_ID': '0'
      }, ['localhost'], 0, 4, {
          xenv.TPU_CHIPS_PER_PROCESS_BOUNDS:
              '1,1,1',
          xenv.TPU_PROCESS_BOUNDS:
              '2,2,1',
          xenv.CLOUD_TPU_TASK_ID:
              '0',
          xenv.TPU_PROCESS_PORT:
              '8476',
          xenv.TPU_PROCESS_ADDRESSES:
              'localhost:8476,localhost:8477,localhost:8478,localhost:8479',
          xenv.TPU_VISIBLE_CHIPS:
              '0',
      }))
  def test_configure_tpu_topology(self, tpu_env, worker_ips, local_rank,
                                  local_world_size, expected):
    with mock.patch.object(tpu, 'get_tpu_env', return_value=tpu_env), \
        mock.patch.object(tpu, 'get_worker_ips', return_value=worker_ips), \
        mock.patch.dict(os.environ, clear=True):

      tpu.configure_topology(local_rank, local_world_size)

      self.assertDictContainsSubset(expected, os.environ)


if __name__ == '__main__':
  absltest.main()
