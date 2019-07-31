#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function


class Worker(object):

  def __init__(self, internal_ip, mach_type, zone, hostname=None):
    self._internal_ip = internal_ip
    self._mach_type = mach_type
    self._zone = zone
    self._hostname = hostname


class Cluster(object):

  def __init__(self, workers):
    self._workers = workers

  def validate(self):
    """Validates the current cluster configuration.

    Raises:
      RuntimeError: If the cluster is misconfigured, this validation will
      raise a error.
    """
    raise NotImplementedError()


class ClusterResolver(object):
  """Cluster Resolver for Client VM and Cloud TPU mesh."""

  def __init__(self,
               tpus,
               vms=None,
               zone=None,
               project=None):
    """Creates a new ClusterResolver object."""

    assert tpus, "TPU name list must not be empty."

    self._tpus = tpus
    self._vms = vms
    self._zone = zone
    self._project = project

  def vm_cluster(self):
    """Gets client VM cluster info.

    The instance group that the current VM belongs to is picked up from
    the GCE instance metadata set of the VM. If a list of VMs was used for
    initializing cluster resolver, we use that instead.

    Returns:
      A Cluster object with the client vm workers information.

    Raises:
      RuntimeError: If the red VM cluster is not healthy or the red VM was
        not created as part of an instance group. If not using an instance
        group, users should just manually configure.
    """
    raise NotImplementedError()

  def tpu_cluster(self):
    """Gets TPU VM cluster info.

    Calls the TPU CLH to get TPU node data and returns list of TPU worker
    VMs internal IP addresses. If zone and project are not specified at
    ClusterResolver init time, we infer these bits from GCE metadata.

    Returns:
      A Cluster object with the tpu workers information.

    Raises:
      RuntimeError: If the TPU DNE or the TPU is in not in HEALTHY state. In the
        sea of single TPUs case, also raises if they're not all in the same
        zone.
    """
    raise NotImplementedError()

