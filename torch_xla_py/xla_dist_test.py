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
        ServiceWorker('10.0.0.0', 'v3-32', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v3-32', 'europe-west4-a'),
        ServiceWorker('10.0.0.2', 'v3-32', 'europe-west4-a'),
        ServiceWorker('10.0.0.3', 'v3-32', 'europe-west4-a'),
    ]
    cluster = Cluster(client_workers, service_workers)
    cluster.validate()  # Does not raise exception

  def test_create_bad_client_workers(self):
    service_workers = [
        ServiceWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
    ]
    client_workers = [
        ClientWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
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
        ServiceWorker('10.0.0.0', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
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
        ServiceWorker('10.0.0.0', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v2-8', 'europe-west4-a'),
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
        ServiceWorker('10.0.0.0', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
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
        ServiceWorker('10.0.0.0', 'v3-32', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v3-32', 'europe-west4-a'),
        ServiceWorker('10.0.0.2', 'v3-32', 'europe-west4-a'),
        ServiceWorker('10.0.0.3', 'v3-32', 'europe-west4-a'),
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


def mock_request_metadata(cls, metadata):
  fake_metadata = {
      'project/project-id': 'fake-project',
      'instance/zone': 'project/fake-project/zone/fake-zone',
      'instance/name': 'fake-ig-a',
  }
  return fake_metadata[metadata]


def build_mock_tpu_service(get_tpu_resp):
  get_node = mock.MagicMock()
  get_node.execute.return_value = get_tpu_resp

  nodes = mock.MagicMock()
  nodes.get.return_value = get_node

  locations = mock.MagicMock()
  locations.nodes.return_value = nodes

  projects = mock.MagicMock()
  projects.locations.return_value = locations

  tpu_service = mock.MagicMock()
  tpu_service.projects.return_value = projects

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
                gen_fake_ig_list_instances_entry('fake-ig-'+c, 'RUNNING')
                for c in 'abcd'
            ],
        },
    }
    instance_resp_map = {
        'fake-ig-'+c: gen_fake_instances_get_entry(
            'fake-ig-'+c, 'n1-standard-16', '10.0.0.'+ip, 'RUNNING')
        for c, ip in zip('abcd', '0123')
    }
    compute_service = build_mock_compute_service(instance_resp_map,
                                                 list_instances_map)
    noop_tpu_service = build_mock_tpu_service({})
    self.mock_discovery.side_effect = build_mock_services_fn(
        compute_service, noop_tpu_service)

    # Act
    cr = ClusterResolver(['fake-tpu'])
    vm_cluster = cr._get_client_workers()

    # Assert
    expected = [
        ClientWorker(internal_ip='10.0.0.'+ip,
                     machine_type='n1-standard-16',
                     zone='fake-zone',
                     hostname='fake-ig-'+c)
        for c, ip in zip('abcd', '0123')
    ]
    self.assertCountEqual(expected, vm_cluster)

  def test_healthy_vm_list_client_cluster(self):
    # Arrange
    list_instances_map = {}
    instance_resp_map = {
        'fake-ig-'+c: gen_fake_instances_get_entry(
            'fake-ig-'+c, 'n1-standard-16', '10.0.0.'+ip, 'RUNNING')
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
    vm_cluster = cr._get_client_workers()

    # Assert
    expected = [
        ClientWorker(internal_ip='10.0.0.'+ip,
                     machine_type='n1-standard-16',
                     zone='fake-zone',
                     hostname='fake-ig-'+c)
        for c, ip in zip('abcd', '0123')
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
                           cr._get_client_workers)

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
                           cr._get_client_workers)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
