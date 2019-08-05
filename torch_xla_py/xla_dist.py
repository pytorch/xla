#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function


class Worker(object):

  def __init__(self, internal_ip, machine_type, zone):
    self._internal_ip = internal_ip
    self._machine_type = machine_type
    self._zone = zone


class ClientWorker(Worker):

  def __init__(self, internal_ip, machine_type, zone, hostname=None):
    super(ClientWorker, self).__init__(internal_ip, machine_type, zone)
    self._hostname = hostname


class ServiceWorker(Worker):
  # Same as base Worker ATM.
  pass


class Cluster(object):

  def __init__(self, client_workers, service_workers,
               check_client_machine_type=True,
               check_service_machine_type=True):
    """Creates a cluster object.

    Args:
      client_workers: a list of ClientWorker objects.
      service_workers: a list of ServiceWorker objects.
      check_client_machine_type: whether to check if client workers all have
        the same machine type.
      check_service_machine_type: whether to check if service workers all have
        the same machine type.
    """
    for client_worker in client_workers:
      if not isinstance(client_worker, ClientWorker):
        raise ValueError(
            'client_workers argument must be a list of ClientWorker')
    for service_worker in service_workers:
      if not isinstance(service_worker, ServiceWorker):
        raise ValueError(
            'service_workers argument must be a list of ServiceWorker')
    self._client_workers = list(client_workers)
    self._service_workers = list(service_workers)
    self._check_client_machine_type = check_client_machine_type
    self._check_service_machine_type = check_service_machine_type

  def validate(self):
    """Validates the current cluster configuration.

    Raises:
      RuntimeError: If the cluster is misconfigured, this validation will
        raise an error. For example, if the VMs are in different zones,
        or not all of the CPU workers have the same size (number of CPU
        cores, RAM size) we raise an exception. For TPUs we similarly
        raise an exception if different zones or machine/accelerator_type.
    """
    if len(self._client_workers) == 0 or len(self._service_workers) == 0:
      raise RuntimeError(
        'Both client_workers and service_workers should not be empty')

    if len(self._client_workers) != len(self._service_workers):
      raise RuntimeError(
          'The client_workers and service_workers must have a 1:1 mapping')

    zones = {worker._zone for worker in self._client_workers}
    zones.update(worker._zone for worker in self._service_workers)
    if len(zones) != 1:
      raise RuntimeError(
          'All workers must be in the same zone, got: {}'.format(zones))

    if self._check_client_machine_type:
      client_machine_types = {
          worker._machine_type for worker in self._client_workers}
      if len(client_machine_types) != 1:
        raise RuntimeError(
            'All client_workers must have the same machine_type, got: {}'.format(
                client_machine_types))

    if self._check_service_machine_type:
      server_machine_types = {
          worker._machine_type for worker in self._service_workers}
      if len(server_machine_types) != 1:
        raise RuntimeError(
            'All service_workers must have the same machine_type, got: {}'.format(
                server_machine_types))


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

  def get_cluster(self):
    """Gets client and server side cluster info.

    If a list of vms is not provided at ClusterResolver crate time the current
    VM's instance group is picked up and we use that to resolve the VM mesh.

    Returns:
      A Cluster object with both client and server mesh configuration.

    Raises:
      RuntimeError: If the VM cluster is not healthy. Also if the TPU
        cluster is not healthy.
    """
    raise NotImplementedError()

