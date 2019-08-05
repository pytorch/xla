"""Tests for xla_dist."""
from __future__ import division
from __future__ import print_function

import unittest
from xla_dist import ClientWorker
from xla_dist import ServiceWorker
from xla_dist import Cluster


class ClusterTest(unittest.TestCase):

  def test_validate_good_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.2', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.3', 'n1-standard-16', 'europe-west4-a',
                     hostname='test'),
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
        ValueError,
        'client_workers argument must be a list of ClientWorker',
        Cluster,
        client_workers, service_workers)

  def test_create_bad_service_workers(self):
    client_workers = [
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a',
                     hostname='test'),
    ]
    self.assertRaisesRegex(
        ValueError,
        'service_workers argument must be a list of ServiceWorker',
        Cluster,
        client_workers, client_workers)

  def test_validate_machine_type_client_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-8', 'europe-west4-a'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v3-8', 'europe-west4-a'),
    ]

    no_check_cluster = Cluster(client_workers, service_workers,
                      check_machine_type=False)
    no_check_cluster.validate()  # Does not raise exception

    check_cluster = Cluster(client_workers, service_workers)
    self.assertRaisesRegex(
        RuntimeError,
        'All client_workers must have the same machine_type',
        check_cluster.validate)

  def test_validate_machine_type_client_cluster(self):
    client_workers = [
        ClientWorker('10.0.0.0', 'n1-standard-16', 'europe-west4-a'),
        ClientWorker('10.0.0.1', 'n1-standard-16', 'europe-west4-a'),
    ]
    service_workers = [
        ServiceWorker('10.0.0.0', 'v3-8', 'europe-west4-a'),
        ServiceWorker('10.0.0.1', 'v2-8', 'europe-west4-a'),
    ]

    no_check_cluster = Cluster(client_workers, service_workers,
                              check_machine_type=False)
    no_check_cluster.validate()  # Does not raise exception

    check_cluster = Cluster(client_workers, service_workers)
    self.assertRaisesRegex(
        RuntimeError,
        'All service_workers must have the same machine_type',
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
    self.assertRaisesRegex(
        RuntimeError,
        'All workers must be in the same zone',
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


class ClusterResolverTest(unittest.TestCase):

  def test_cluster(self):
    pass


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
