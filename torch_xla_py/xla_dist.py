#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function

import abc


class ClientWorker(object):

  def __init__(self,
               internal_ip,
               machine_type,
               zone,
               hostname=None):
    self._internal_ip = internal_ip
    self._machine_type = machine_type
    self._zone = zone
    self._hostname = hostname


class ServiceWorker(object):

  def __init__(self,
               internal_ip,
               accelerator_type,
               zone):
    self._internal_ip = internal_ip
    self._accelerator_type = accelerator_type
    self._zone = zone


class BaseCluster(abc.ABC):

  @abc.abstractmethod
  def validate(cls, workers):
    """Validates the current cluster configuration.

    Raises:
      RuntimeError: If the cluster is misconfigured, this validation will
      raise a error.
    """
    raise NotImplementedError()


class ClientCluster(BaseCluster):

  def __init__(self, workers):
    for worker in workers:
      if not isinstance(worker, ClientWorker):
        raise ValueError('ClientCluster workers must be ClientWorker.')
    self._workers = workers

  def validate(self):
    """Validates the client workers configuration.

    Raises:
      RuntimeError: If the client cluster is misconfigured. For example, if
        the VMs are in different zones, or not all of the CPU workers have
        the same size (number of CPU cores, RAM size) we raise an exception
        as this would negatively and significantly affect performance.
    """
    raise NotImplementedError()


class ServiceCluster(BaseCluster):

  def __init__(self, workers):
    for worker in workers:
      if not isinstance(worker, ServiceWorker):
        raise ValueError('ServiceCluster workers must be ServiceWorker.')
    self._workers = workers

  def validate(self):
    """Validates the service cluster configuration.

    Raises:
      RuntimeError: If the service cluster is misconfigured. For example, if the
        TPUs are in different zones, or the TPUs are different versions we
        raise an exception as this silently impacts performance.
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
      A ClientCluster object with the client vm workers information.

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
      A ServiceCluster object with the tpu workers information.

    Raises:
      RuntimeError: If the TPU DNE or the TPU is in not in HEALTHY state. In the
        sea of single TPUs case, also raises if they're not all in the same
        zone.
    """
    raise NotImplementedError()

