"""Tests for xla_dist."""
from __future__ import division
from __future__ import print_function

import cloud_tpu_client
import uuid
import unittest
from unittest import mock

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from torch_xla.distributed.cluster import Cluster
from torch_xla.distributed.cluster import ClusterResolver
from torch_xla.distributed.worker import ClientWorker
from torch_xla.distributed.worker import ServiceWorker

PROJECT_ZONE_PREFIX = ('https://www.googleapis.com/compute/v1/'
                       'projects/fake-project/zones/fake-zone')
TPUVM_HOSTNAME_PREFIX = 't1v-n-5d9c8fb2-w-'


class ClusterTest(unittest.TestCase):

  def test_validate_good_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.2', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker(
            '10.0.0.3', 'n1-standard-16', 'europe-west4-a', hostname='test'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.1', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.2', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.3', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
    ]
    cluster = Cluster(
        client_workers, service_workers, client_master_ip='10.0.0.0')
    cluster.validate()  # Does not raise exception

  def test_create_bad_client_workers(self):
    service_workers = [
        ServiceWorker('10.0.0.1', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
    ]
    client_workers = [
        ClientWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
    ]
    self.assertRaisesRegex(
        ValueError,
        'client_workers argument must be a list of ClientWorker',
        Cluster,
        client_workers,
        service_workers,
        client_master_ip='10.0.0.1')

  def test_create_bad_service_workers(self):
    client_workers = [
        ClientWorker(
            '10.0.0.1', 'n1-standard-16', 'europe-west4-a', hostname='test'),
    ]
    self.assertRaisesRegex(
        ValueError,
        'service_workers argument must be a list of ServiceWorker',
        Cluster,
        client_workers,
        client_workers,
        client_master_ip='10.0.0.1')

  def test_validate_machine_type_client_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-8', 'europe-west4-a'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.1', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
    ]

    no_check_cluster = Cluster(
        client_workers,
        service_workers,
        check_client_machine_type=False,
        client_master_ip='10.0.0.0')
    no_check_cluster.validate()  # Does not raise exception

    check_cluster = Cluster(
        client_workers, service_workers, client_master_ip='10.0.0.0')
    self.assertRaisesRegex(
        RuntimeError, 'All client_workers must have the same machine_type',
        check_cluster.validate)

  def test_validate_machine_type_service_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.1', '8470', 'v2-8', 'europe-west4-a',
                      'pytorch-0.2'),
    ]

    no_check_cluster = Cluster(
        client_workers,
        service_workers,
        check_service_machine_type=False,
        client_master_ip='10.0.0.0')
    no_check_cluster.validate()  # Does not raise exception

    check_cluster = Cluster(
        client_workers, service_workers, client_master_ip='10.0.0.0')
    self.assertRaisesRegex(
        RuntimeError, 'All service_workers must have the same machine_type',
        check_cluster.validate)

  def test_validate_bad_zone_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'us-central1-b'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.1', '8470', 'v3-8', 'europe-west4-a',
                      'pytorch-0.2'),
    ]
    cluster = Cluster(
        client_workers, service_workers, client_master_ip='10.0.0.0')
    self.assertRaisesRegex(RuntimeError, 'All workers must be in the same zone',
                           cluster.validate)

  def test_validate_diff_num_workers(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.2', 'n1-standard-16', 'europe-west4-a'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.1', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.2', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.3', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
    ]
    cluster = Cluster(
        client_workers, service_workers, client_master_ip='10.0.0.0')
    self.assertRaisesRegex(
        RuntimeError,
        'The client_workers and service_workers must have a 1:1 mapping',
        cluster.validate)

  def test_validate_empty_workers(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a')
    ]
    cluster = Cluster(client_workers, [], client_master_ip='10.0.0.0')
    self.assertRaisesRegex(
        RuntimeError,
        'Both client_workers and service_workers should not be empty',
        cluster.validate)

  def test_validate_diff_runtime_versions(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.2', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.3', 'n1-standard-16', 'europe-west4-a'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.1'),
        ServiceWorker('10.0.0.1', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
        ServiceWorker('10.0.0.2', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.1'),
        ServiceWorker('10.0.0.3', '8470', 'v3-32', 'europe-west4-a',
                      'pytorch-0.2'),
    ]
    cluster = Cluster(
        client_workers, service_workers, client_master_ip='10.0.0.0')
    self.assertRaisesRegex(
        RuntimeError,
        'All service workers must have the same runtime_version.*',
        cluster.validate)


def mock_request_metadata(metadata):
  fake_metadata = {
      'project/project-id': 'fake-project',
      'instance/zone': 'project/fake-project/zones/fake-zone',
      'instance/name': 'fake-ig-a',
      'instance/network-interfaces/0/ip': '10.0.0.0',
      # Adding this field to prevent crashing when ClusterResolver querying this
      # metadata to identify the TPUVM case.
      'instance/attributes/accelerator-type': '',
  }
  return fake_metadata[metadata]


def mock_request_tpuvm_metadata(metadata):
  fake_metadata = {
      'project/project-id': 'fake-project',
      'instance/zone': 'project/fake-project/zones/fake-zone',
      'instance/name': TPUVM_HOSTNAME_PREFIX + '0',
      'instance/network-interfaces/0/ip': '10.1.0.0',
      'instance/attributes/accelerator-type': 'v3-32',
  }
  return fake_metadata[metadata]


def mock_ip_to_hostname_mapping(tpu_name, zone, num_vm):
  ip_to_hostname_map = {}
  for index in range(num_vm):
    ip_to_hostname_map[f'10.1.0.{index}'] = f'{TPUVM_HOSTNAME_PREFIX}{index}'
  return ip_to_hostname_map


def build_mock_cloud_tpu_client_library(tpu_map):

  def mock_cloud_tpu_client_constructor(*args, **kwargs):
    # Patch to mock cloud_tpu_client.Client.__init__ method.
    tpu_name = kwargs['tpu']
    tpu_dict = tpu_map[tpu_name]
    ctc = mock.MagicMock()
    ctc.name.return_value = tpu_name
    ctc.state.return_value = tpu_dict.get('state')
    ctc.health.return_value = tpu_dict.get('health')
    ctc.runtime_version.return_value = tpu_dict.get('runtime_version')
    ctc.accelerator_type.return_value = tpu_dict.get('accelerator_type')
    ctc.network_endpoints.return_value = tpu_dict.get('network_endpoints')
    # TODO: add a api to get the tpu api version directly
    ctc._get_tpu_property.return_value = tpu_dict.get('api_version')
    ctc._full_name.return_value = \
      f'projects/fake-project/locations/fake-zone/nodes/{tpu_name}'
    return ctc

  return mock_cloud_tpu_client_constructor


def build_mock_compute_service(get_instance_map, list_instances_map):
  # Instances mock
  def get_instance_fn(*args, **kwargs):
    resp = get_instance_map[kwargs['instance']]
    get_instance = mock.MagicMock()
    get_instance.execute.return_value = resp
    get_instance.resumable = None
    return get_instance

  instances = mock.MagicMock()
  instances.get.side_effect = get_instance_fn

  # Instance groups mock
  def list_instances_fn(*args, **kwargs):
    resp = list_instances_map[kwargs['instanceGroup']]
    list_instances = mock.MagicMock()
    list_instances.execute.return_value = resp
    return list_instances

  instance_groups = mock.MagicMock()
  instance_groups.listInstances.side_effect = list_instances_fn

  # Compute service mock
  compute_service = mock.MagicMock()
  compute_service.instances.return_value = instances
  compute_service.instanceGroups.return_value = instance_groups
  compute_service.new_batch_http_request.return_value = build_mock_batch_call()

  return compute_service


def build_mock_services_fn(mock_compute_service):

  def mock_google_services(serviceName, version, **kwargs):
    if serviceName == 'compute':
      return mock_compute_service
    else:
      raise RuntimeError(f'Service name "{serviceName}" is not mocked.')

  return mock_google_services


def build_mock_batch_call():
  batcher = mock.MagicMock()

  def build_execute_requests_fn(call_list):

    def execute_requests(*args):
      del args
      for args, _ in call_list:
        req, callback = args
        resp = None
        exception = None
        try:
          resp = req.execute()
        except e:
          exception = e
        callback(uuid.uuid4(), resp, exception)

    return execute_requests

  batcher.execute.side_effect = build_execute_requests_fn(
      batcher.add.call_args_list)
  return batcher


def gen_fake_instances_get_entry(instance_name, machine_type, internal_ip,
                                 status):
  return {
      'machineType': f'{PROJECT_ZONE_PREFIX}/machineTypes/{machine_type}',
      'metadata': {
          'fingerprint': 'abc',
          'items': [{
              'key':
                  'instance-template',
              'value': ('projects/123456789012/global/'
                        'instanceTemplates/fake-ig-template'),
          }, {
              'key':
                  'created-by',
              'value': ('projects/123456789012/zones/fake-zone/'
                        'instanceGroupManagers/fake-ig'),
          }],
          'kind': 'compute#metadata',
      },
      'selfLink': f'{PROJECT_ZONE_PREFIX}/instances/{instance_name}',
      'networkInterfaces': [{
          'networkIP': internal_ip,
      }],
      'status': status,
      'zone': PROJECT_ZONE_PREFIX,
  }


def gen_fake_ig_list_instances_entry(instance_name, status):
  return {
      'instance': f'{PROJECT_ZONE_PREFIX}/instances/{instance_name}',
      'status': status,
  }


class ClusterResolverTest(unittest.TestCase):

  def setUp(self):
    super(ClusterResolverTest, self).setUp()
    self.addCleanup(mock.patch.stopall)
    mock.patch.object(ClusterResolver, 'get_instance_metadata',
                      mock_request_metadata).start()
    mock.patch.object(ClusterResolver, '_get_internal_ip_to_hostname_mapping',
                      mock_ip_to_hostname_mapping).start()
    mock.patch.object(GoogleCredentials, 'get_application_default',
                      lambda *args, **kwargs: None).start()
    self.mock_discovery = mock.patch.object(
        discovery, 'build', autospec=True).start()
    self.mock_ctc = mock.patch.object(
        cloud_tpu_client, 'Client', autospec=True).start()

  def test_bad_empty_tpu_constructor(self):
    tpus = ''
    self.assertRaisesRegex(ValueError, 'tpu must be a non-empty string',
                           ClusterResolver, tpus)

  def test_bad_none_tpu_constructor(self):
    tpus = None
    self.assertRaisesRegex(ValueError, 'tpu must be a non-empty string',
                           ClusterResolver, tpus)

  def test_bad_vm_constructor(self):
    tpus = ['fake-tpu']
    vms = {'abc'}
    self.assertRaisesRegex(ValueError,
                           'vms must be a non-empty list if provided',
                           ClusterResolver, tpus, vms)

  def test_healthy_instance_group_client_cluster(self):
    # Arrange
    list_instances_map = {
        'fake-ig': {
            'kind':
                'compute#instanceGroupsListInstances',
            'items': [
                gen_fake_ig_list_instances_entry('fake-ig-' + c, 'RUNNING')
                for c in 'abcd'
            ],
        },
    }
    instance_resp_map = {
        'fake-ig-' + c:
        gen_fake_instances_get_entry('fake-ig-' + c, 'n1-standard-16',
                                     '10.0.0.' + ip, 'RUNNING')
        for c, ip in zip('abcd', '0123')
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    self.mock_discovery.side_effect = build_mock_services_fn(compute_service)

    # Act
    cr = ClusterResolver(['fake-tpu'])
    vm_cluster = cr.get_client_workers()

    # Assert
    expected = [
        ClientWorker(
            internal_ip='10.0.0.' + ip,
            machine_type='n1-standard-16',
            zone='fake-zone',
            hostname='fake-ig-' + c) for c, ip in zip('abcd', '0123')
    ]
    self.assertCountEqual(expected, vm_cluster)

  def test_healthy_vm_list_client_cluster(self):
    # Arrange
    list_instances_map = {}
    instance_resp_map = {
        'fake-ig-' + c:
        gen_fake_instances_get_entry('fake-ig-' + c, 'n1-standard-16',
                                     '10.0.0.' + ip, 'RUNNING')
        for c, ip in zip('abcd', '0123')
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    self.mock_discovery.side_effect = build_mock_services_fn(compute_service)

    # Act
    vms = ['fake-ig-a', 'fake-ig-b', 'fake-ig-c', 'fake-ig-d']
    cr = ClusterResolver(['fake-tpu'], vms=vms)
    vm_cluster = cr.get_client_workers()

    # Assert
    expected = [
        ClientWorker(
            internal_ip='10.0.0.' + ip,
            machine_type='n1-standard-16',
            zone='fake-zone',
            hostname='fake-ig-' + c) for c, ip in zip('abcd', '0123')
    ]
    self.assertCountEqual(expected, vm_cluster)

  def test_empty_instance_group_client_cluster(self):
    list_instances_map = {
        'fake-ig': {
            'kind': 'compute#instanceGroupsListInstances',
            'items': [],
        },
    }
    instance_resp_map = {
        'fake-ig-a':
            gen_fake_instances_get_entry('fake-ig-a', 'n1-standard-16',
                                         '10.0.0.0', 'RUNNING'),
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    self.mock_discovery.side_effect = build_mock_services_fn(compute_service)

    # Act
    cr = ClusterResolver(['fake-tpu'])

    # Assert
    self.assertRaisesRegex(RuntimeError, '.*vms is empty in instance group.*',
                           cr.get_client_workers)

  def test_unhealthy_client_cluster(self):
    # Arrange
    list_instances_map = {
        'fake-ig': {
            'kind':
                'compute#instanceGroupsListInstances',
            'items': [
                gen_fake_ig_list_instances_entry('fake-ig-a', 'RUNNING'),
                gen_fake_ig_list_instances_entry('fake-ig-b', 'PROVISIONING'),
                gen_fake_ig_list_instances_entry('fake-ig-c', 'RUNNING'),
                gen_fake_ig_list_instances_entry('fake-ig-d', 'RUNNING'),
            ],
        },
    }
    instance_resp_map = {
        'fake-ig-a':
            gen_fake_instances_get_entry('fake-ig-a', 'n1-standard-16',
                                         '10.0.0.0', 'RUNNING'),
        'fake-ig-b':
            gen_fake_instances_get_entry('fake-ig-b', 'n1-standard-16',
                                         '10.0.0.1', 'PROVISIONING'),
        'fake-ig-c':
            gen_fake_instances_get_entry('fake-ig-c', 'n1-standard-16',
                                         '10.0.0.2', 'RUNNING'),
        'fake-ig-d':
            gen_fake_instances_get_entry('fake-ig-d', 'n1-standard-16',
                                         '10.0.0.3', 'RUNNING'),
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    self.mock_discovery.side_effect = build_mock_services_fn(compute_service)

    # Act
    cr = ClusterResolver(['fake-tpu'])

    # Assert
    self.assertRaisesRegex(RuntimeError,
                           'Instance fake-ig-b is not running yet.*',
                           cr.get_client_workers)

  def test_healthy_pod_service_cluster(self):
    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-32',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            } for ip in range(4)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    service_workers = cr.get_tpu_workers()

    expected = [
        ServiceWorker(
            internal_ip=f'10.0.0.{ip}',
            port='8470',
            machine_type='v3-32',
            zone='fake-zone',
            runtime_version='pytorch-nightly',
            tpu='fake-pod') for ip in range(4)
    ]
    self.assertCountEqual(expected, service_workers)

  def test_healthy_sea_service_cluster(self):
    noop_compute_service = build_mock_compute_service({}, {})
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service)
    tpu_map = {
        f'fake-tpu-{ip}': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-8',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            }],
        } for ip in range(256)
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    service_workers = cr.get_tpu_workers()

    expected = [
        ServiceWorker(
            internal_ip=f'10.0.0.{ip}',
            port='8470',
            machine_type='v3-8',
            zone='fake-zone',
            runtime_version='pytorch-nightly',
            tpu=f'fake-tpu-{ip}') for ip in range(256)
    ]
    self.assertCountEqual(expected, service_workers)

  def test_unhealthy_pod_service_cluster(self):
    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'health':
                'UNHEALTHY_TENSORFLOW',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-128',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            } for ip in range(16)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(RuntimeError, 'TPU fake-pod is not HEALTHY yet.*',
                           cr.get_tpu_workers)

  def test_non_ready_sea_service_cluster(self):
    noop_compute_service = build_mock_compute_service({}, {})
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service)

    tpu_map = {
        f'fake-tpu-{ip}': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-8',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            }],
        } for ip in range(3)
    }
    tpu_map['fake-tpu-3'] = {
        'state': 'CREATING',
        'runtime_version': 'pytorch-nightly',
        'accelerator_type': 'v3-8',
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(RuntimeError, 'TPU fake-tpu-3 is not READY yet.*',
                           cr.get_tpu_workers)

  def test_unknown_health_pod_service_cluster(self):
    noop_compute_service = build_mock_compute_service({}, {})
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service)
    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-32',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            } for ip in range(4)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(RuntimeError, 'TPU fake-pod is not HEALTHY yet.*',
                           cr.get_tpu_workers)

  def test_healthy_cluster(self):
    list_instances_map = {
        'fake-ig': {
            'kind':
                'compute#instanceGroupsListInstances',
            'items': [
                gen_fake_ig_list_instances_entry('fake-ig-' + c, 'RUNNING')
                for c in 'abcd'
            ],
        },
    }
    instance_resp_map = {
        'fake-ig-' + c:
        gen_fake_instances_get_entry('fake-ig-' + c, 'n1-standard-16',
                                     '10.0.0.' + ip, 'RUNNING')
        for c, ip in zip('abcd', '0123')
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    self.mock_discovery.side_effect = build_mock_services_fn(compute_service)

    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-32',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            } for ip in range(4)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    cluster = cr.get_cluster()

    expected_client_workers = [
        ClientWorker(
            internal_ip='10.0.0.' + ip,
            machine_type='n1-standard-16',
            zone='fake-zone',
            hostname='fake-ig-' + c) for c, ip in zip('abcd', '0123')
    ]
    expected_service_workers = [
        ServiceWorker(
            internal_ip=f'10.0.0.{ip}',
            port='8470',
            machine_type='v3-32',
            zone='fake-zone',
            runtime_version='pytorch-nightly',
            tpu='fake-pod') for ip in range(4)
    ]
    expected = Cluster(
        expected_client_workers,
        expected_service_workers,
        client_master_ip='10.0.0.0')
    self.assertEqual(expected, cluster)

  def test_healthy_remote_coordinator(self):
    noop_compute_service = build_mock_compute_service({}, {})
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service)

    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'v2-nightly',
            'accelerator_type':
                'v3-32',
            'api_version':
                'V2_ALPHA1',
            'network_endpoints': [{
                'ipAddress': f'10.1.0.{index}',
                'port': '8470',
            } for index in range(4)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    cluster = cr.get_cluster()

    expected_client_workers = [
        ClientWorker(
            internal_ip=f'10.1.0.{index}',
            machine_type='v3-32',
            zone='fake-zone',
            hostname=f'{TPUVM_HOSTNAME_PREFIX}{index}') for index in range(4)
    ]
    expected_service_workers = [
        ServiceWorker(
            internal_ip=f'10.1.0.{ip}',
            port='8470',
            machine_type='v3-32',
            zone='fake-zone',
            runtime_version='v2-nightly',
            tpu='fake-pod') for ip in range(4)
    ]
    expected = Cluster(
        expected_client_workers,
        expected_service_workers,
        client_master_ip='10.1.0.0')
    self.assertEqual(expected, cluster)

  def test_healthy_tpuvm_cluster(self):
    # Using TPUVM flavor of metadata.
    mock.patch.object(ClusterResolver, 'get_instance_metadata',
                      mock_request_tpuvm_metadata).start()
    noop_compute_service = build_mock_compute_service({}, {})
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service)

    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'v2-nightly',
            'accelerator_type':
                'v3-32',
            'api_version':
                'V2_ALPHA1',
            'network_endpoints': [{
                'ipAddress': f'10.1.0.{index}',
                'port': '8470',
            } for index in range(4)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    cluster = cr.get_cluster()

    expected_client_workers = [
        ClientWorker(
            internal_ip=f'10.1.0.{index}',
            machine_type='v3-32',
            zone='fake-zone',
            hostname=f'{TPUVM_HOSTNAME_PREFIX}{index}') for index in range(4)
    ]
    expected_service_workers = [
        ServiceWorker(
            internal_ip=f'10.1.0.{ip}',
            port='8470',
            machine_type='v3-32',
            zone='fake-zone',
            runtime_version='v2-nightly',
            tpu='fake-pod') for ip in range(4)
    ]
    expected = Cluster(
        expected_client_workers,
        expected_service_workers,
        client_master_ip='10.1.0.0')
    self.assertEqual(expected, cluster)
    mock.patch.object(ClusterResolver, 'get_instance_metadata',
                      mock_request_metadata).start()

  def test_bad_cluster(self):
    list_instances_map = {
        'fake-ig': {
            'kind':
                'compute#instanceGroupsListInstances',
            'items': [
                gen_fake_ig_list_instances_entry('fake-ig-' + c, 'RUNNING')
                for c in 'abc'
            ],
        },
    }
    instance_resp_map = {
        'fake-ig-' + c:
        gen_fake_instances_get_entry('fake-ig-' + c, 'n1-standard-16',
                                     '10.0.0.' + ip, 'RUNNING')
        for c, ip in zip('abcd', '0123')
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    self.mock_discovery.side_effect = build_mock_services_fn(compute_service)

    tpu_map = {
        'fake-pod': {
            'state':
                'READY',
            'health':
                'HEALTHY',
            'runtime_version':
                'pytorch-nightly',
            'accelerator_type':
                'v3-32',
            'network_endpoints': [{
                'ipAddress': f'10.0.0.{ip}',
                'port': '8470'
            } for ip in range(4)],
        }
    }
    self.mock_ctc.side_effect = build_mock_cloud_tpu_client_library(tpu_map)

    tpus = list(tpu_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(
        RuntimeError,
        'The client_workers and service_workers must have a 1:1 mapping',
        cr.get_cluster)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
