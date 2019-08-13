"""Tests for xla_dist."""
from __future__ import division
from __future__ import print_function

import uuid
import unittest
from unittest import mock

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from xla_dist import Cluster
from xla_dist import ClusterResolver
from xla_dist import ClientWorker
from xla_dist import ServiceWorker

PROJECT_ZONE_PREFIX = ('https://www.googleapis.com/compute/v1/'
                       'projects/fake-project/zones/fake-zone')


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
    cluster = Cluster(client_workers, service_workers)
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
        ValueError, 'client_workers argument must be a list of ClientWorker',
        Cluster, client_workers, service_workers)

  def test_create_bad_service_workers(self):
    client_workers = [
        ClientWorker(
            '10.0.0.1', 'n1-standard-16', 'europe-west4-a', hostname='test'),
    ]
    self.assertRaisesRegex(
        ValueError, 'service_workers argument must be a list of ServiceWorker',
        Cluster, client_workers, client_workers)

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
        client_workers, service_workers, check_client_machine_type=False)
    no_check_cluster.validate()  # Does not raise exception

    check_cluster = Cluster(client_workers, service_workers)
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
        client_workers, service_workers, check_service_machine_type=False)
    no_check_cluster.validate()  # Does not raise exception

    check_cluster = Cluster(client_workers, service_workers)
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
    cluster = Cluster(client_workers, service_workers)
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
    cluster = Cluster(client_workers, service_workers)
    self.assertRaisesRegex(
        RuntimeError,
        'The client_workers and service_workers must have a 1:1 mapping',
        cluster.validate)

  def test_validate_empty_workers(self):
    cluster = Cluster([], [])
    self.assertRaisesRegex(
        RuntimeError,
        'Both client_workers and service_workers should not be empty',
        cluster.validate)

  def test_validate_diff_sw_versions(self):
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
    cluster = Cluster(client_workers, service_workers)
    self.assertRaisesRegex(
        RuntimeError, 'All service workers must have the same sw_version.*',
        cluster.validate)


def mock_request_metadata(cls, metadata):
  fake_metadata = {
      'project/project-id': 'fake-project',
      'instance/zone': 'project/fake-project/zones/fake-zone',
      'instance/name': 'fake-ig-a',
  }
  return fake_metadata[metadata]


def build_mock_tpu_service(get_tpu_resp):

  def get_tpu_fn(*args, **kwargs):
    node_name = ClusterResolver._parse_resource_url(kwargs['name'], 'nodes')
    resp = get_tpu_resp[node_name]
    get_node = mock.MagicMock()
    get_node.execute.return_value = resp
    return get_node

  nodes = mock.MagicMock()
  nodes.get.side_effect = get_tpu_fn

  locations = mock.MagicMock()
  locations.nodes.return_value = nodes

  projects = mock.MagicMock()
  projects.locations.return_value = locations

  tpu_service = mock.MagicMock()
  tpu_service.projects.return_value = projects
  tpu_service.new_batch_http_request.return_value = build_mock_batch_call()

  return tpu_service


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


def build_mock_services_fn(mock_compute_service, mock_tpu_service):

  def mock_google_services(serviceName, version, **kwargs):
    if serviceName == 'compute':
      return mock_compute_service
    elif serviceName == 'tpu':
      return mock_tpu_service
    else:
      raise RuntimeError('Service name "{}" is not mocked.'.format(serviceName))

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
      'machineType':
          '{}/machineTypes/{}'.format(PROJECT_ZONE_PREFIX, machine_type),
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
      'selfLink':
          '{}/instances/{}'.format(PROJECT_ZONE_PREFIX, instance_name),
      'networkInterfaces': [{
          'networkIP': internal_ip,
      }],
      'status':
          status,
      'zone':
          PROJECT_ZONE_PREFIX,
  }


def gen_fake_ig_list_instances_entry(instance_name, status):
  return {
      'instance': '{}/instances/{}'.format(PROJECT_ZONE_PREFIX, instance_name),
      'status': status,
  }


def gen_fake_tpu_entry(accelerator_type,
                       internal_ips,
                       name,
                       state,
                       sw_version,
                       health=None):
  resp = {
      'acceleratorType': accelerator_type,
      'ipAddress': internal_ips[0],
      'name': 'projects/fake-project/locations/fake-zone/nodes/{}'.format(name),
      'networkEndpoints': [{
          'ipAddress': internal_ip,
          'port': 8470
      } for internal_ip in internal_ips],
      'state': state,
      'tensorflowVersion': sw_version,
  }
  if health is not None:
    resp['health'] = health
  return resp


class ClusterResolverTest(unittest.TestCase):

  def setUp(self):
    super(ClusterResolverTest, self).setUp()
    self.addCleanup(mock.patch.stopall)
    mock.patch.object(ClusterResolver, '_get_instance_metadata',
                      mock_request_metadata).start()
    mock.patch.object(GoogleCredentials, 'get_application_default',
                      lambda *args, **kwargs: None).start()
    self.mock_discovery = mock.patch.object(
        discovery, 'build', autospec=True).start()

  def test_bad_empty_tpu_constructor(self):
    tpus = []
    self.assertRaisesRegex(ValueError, 'tpus must be a non-empty list',
                           ClusterResolver, tpus)

  def test_bad_none_tpu_constructor(self):
    tpus = None
    self.assertRaisesRegex(ValueError, 'tpus must be a non-empty list',
                           ClusterResolver, tpus)

  def test_bad_empty_vm_constructor(self):
    tpus = ['fake-tpu']
    vms = []
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
    noop_tpu_service = build_mock_tpu_service({})
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, noop_tpu_service)

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
    noop_tpu_service = build_mock_tpu_service({})
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, noop_tpu_service)

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
    noop_tpu_service = build_mock_tpu_service({})
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, noop_tpu_service)

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
    noop_tpu_service = build_mock_tpu_service({})
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, noop_tpu_service)

    # Act
    cr = ClusterResolver(['fake-tpu'])

    # Assert
    self.assertRaisesRegex(RuntimeError,
                           'Instance fake-ig-b is not running yet.*',
                           cr.get_client_workers)

  def test_healthy_pod_service_cluster(self):
    tpu_resp_map = {
        'fake-pod':
            gen_fake_tpu_entry(
                'v3-2048', ['10.0.0.{}'.format(ip) for ip in range(256)],
                'fake-pod',
                'READY',
                'nightly',
                health='HEALTHY'),
    }
    noop_compute_service = build_mock_compute_service({}, {})
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
    cr = ClusterResolver(tpus)
    service_workers = cr.get_service_workers()

    expected = [
        ServiceWorker(
            internal_ip='10.0.0.{}'.format(ip),
            port='8470',
            machine_type='v3-2048',
            zone='fake-zone',
            sw_version='nightly') for ip in range(256)
    ]
    self.assertCountEqual(expected, service_workers)

  def test_healthy_sea_service_cluster(self):
    tpu_resp_map = {
        'fake-tpu-{}'.format(ip): gen_fake_tpu_entry(
            'v3-8', ['10.0.0.{}'.format(ip)],
            'fake-tpu-{}'.format(ip),
            'READY',
            'nightly',
            health='HEALTHY') for ip in range(256)
    }
    noop_compute_service = build_mock_compute_service({}, {})
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
    cr = ClusterResolver(tpus)
    service_workers = cr.get_service_workers()

    expected = [
        ServiceWorker(
            internal_ip='10.0.0.{}'.format(ip),
            port='8470',
            machine_type='v3-8',
            zone='fake-zone',
            sw_version='nightly') for ip in range(256)
    ]
    self.assertCountEqual(expected, service_workers)

  def test_unhealthy_pod_service_cluster(self):
    tpu_resp_map = {
        'fake-pod':
            gen_fake_tpu_entry(
                'v3-128', ['10.0.0.{}'.format(ip) for ip in range(16)],
                'fake-pod',
                'READY',
                'nightly',
                health='UNHEALTHY_TENSORFLOW'),
    }
    noop_compute_service = build_mock_compute_service({}, {})
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(RuntimeError, 'TPU fake-pod is not HEALTHY yet.*',
                           cr.get_service_workers)

  def test_non_ready_sea_service_cluster(self):
    tpu_resp_map = {
        'fake-tpu-0':
            gen_fake_tpu_entry(
                'v3-8', ['10.0.0.0'],
                'fake-tpu-0',
                'READY',
                'nightly',
                health='HEALTHY'),
        'fake-tpu-1':
            gen_fake_tpu_entry(
                'v3-8', ['10.0.0.1'],
                'fake-tpu-1',
                'READY',
                'nightly',
                health='HEALTHY'),
        'fake-tpu-2':
            gen_fake_tpu_entry('v3-8', ['10.0.0.2'], 'fake-tpu-2', 'CREATING',
                               'nightly'),
        'fake-tpu-3':
            gen_fake_tpu_entry(
                'v3-8', ['10.0.0.3'],
                'fake-tpu-3',
                'READY',
                'nightly',
                health='HEALTHY'),
    }
    noop_compute_service = build_mock_compute_service({}, {})
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(RuntimeError, 'TPU fake-tpu-2 is not READY yet.*',
                           cr.get_service_workers)

  def test_unknown_health_pod_service_cluster(self):
    tpu_resp_map = {
        'fake-pod':
            gen_fake_tpu_entry(
                'v3-32', ['10.0.0.{}'.format(ip) for ip in range(4)],
                'fake-pod',
                'READY',
                'nightly'),
    }
    noop_compute_service = build_mock_compute_service({}, {})
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        noop_compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(RuntimeError, 'TPU fake-pod is not HEALTHY yet.*',
                           cr.get_service_workers)

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

    tpu_resp_map = {
        'fake-pod':
            gen_fake_tpu_entry(
                'v3-32', ['10.0.0.{}'.format(ip) for ip in range(4)],
                'fake-pod',
                'READY',
                'nightly',
                health='HEALTHY'),
    }
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
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
            internal_ip='10.0.0.{}'.format(ip),
            port='8470',
            machine_type='v3-32',
            zone='fake-zone',
            sw_version='nightly') for ip in range(4)
    ]
    expected = Cluster(expected_client_workers, expected_service_workers)
    self.assertEqual(expected, cluster)

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

    tpu_resp_map = {
        'fake-pod':
            gen_fake_tpu_entry(
                'v3-32', ['10.0.0.{}'.format(ip) for ip in range(4)],
                'fake-pod',
                'READY',
                'nightly',
                health='HEALTHY'),
    }
    tpu_service = build_mock_tpu_service(tpu_resp_map)
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, tpu_service)

    tpus = list(tpu_resp_map.keys())
    cr = ClusterResolver(tpus)
    self.assertRaisesRegex(
        RuntimeError,
        'The client_workers and service_workers must have a 1:1 mapping',
        cr.get_cluster)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
