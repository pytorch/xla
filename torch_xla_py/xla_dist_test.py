"""Tests for xla_dist."""
from __future__ import division
from __future__ import print_function

import unittest
import xla_dist

class ClusterTest(unittest.TestCase):

  def test_validate_service_cluster(self):
    pass

  def test_validate_client_cluster(self):
    pass


class ClusterResolverTest(unittest.TestCase):

  def test_vm_cluster(self):
    pass

  def test_tpu_cluster(self):
    pass


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
