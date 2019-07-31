#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function

import requests

try:
  from googleapiclient import discovery
  from googleapiclient.http import BatchHttpRequest
  from oauth2client.client import GoogleCredentials
except ImportError:
  raise ImportError('googleapiclient and oauth2client must be installed '
                    'before using the xla_dist. Execute: '
                    '`pip install --upgrade google-api-python-client` '
                    'and `pip install --upgrade oauth2client` to '
                    'install with pip.')

_GCE_METADATA_ENDPOINT = 'http://metadata.google.internal'


class Worker(object):

  def __init__(self, internal_ip, machine_type, zone):
    self._internal_ip = internal_ip
    self._machine_type = machine_type
    self._zone = zone


class ClientWorker(Worker):

  def __init__(self, internal_ip, machine_type, zone, hostname=None):
    super(ClientWorker, self).__init__(internal_ip, machine_type, zone)
    self._hostname = hostname

  def __repr__(self):
    return ('ClientWorker({internal_ip}, {machine_type}, {zone},'
            ' {hostname})').format(
                internal_ip=self._internal_ip,
                machine_type=self._machine_type,
                zone=self._zone,
                hostname=self._hostname)

  def __eq__(self, other):
    return (self._internal_ip == other._internal_ip and
            self._machine_type == other._machine_type and
            self._zone == other._zone and self._hostname == other._hostname)

  def __ne__(self, other):
    return not self.__eq__(self, other)

  def __hash__(self):
    return hash(repr(self))


class ServiceWorker(Worker):
  # Same as base Worker ATM.
  pass


class Cluster(object):

  def __init__(self,
               client_workers,
               service_workers,
               check_client_machine_type=True,
               check_service_machine_type=True):
    """Creates a cluster object.

    Args:
      client_workers: a list of ClientWorker objects.
      service_workers: a list of ServiceWorker objects.
      check_client_machine_type: whether to check if client workers all have the
        same machine type.
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
          worker._machine_type for worker in self._client_workers
      }
      if len(client_machine_types) != 1:
        raise RuntimeError(
            'All client_workers must have the same machine_type, got: {}'
            .format(client_machine_types))

    if self._check_service_machine_type:
      server_machine_types = {
          worker._machine_type for worker in self._service_workers
      }
      if len(server_machine_types) != 1:
        raise RuntimeError(
            'All service_workers must have the same machine_type, got: {}'
            .format(server_machine_types))


class ClusterResolver(object):
  """Cluster Resolver for Client VM and Cloud TPU mesh."""

  @classmethod
  def _get_instance_metadata(cls, metadata):
    response = requests.get(
        '{}/computeMetadata/v1/{}'.format(_GCE_METADATA_ENDPOINT, metadata),
        headers={'Metadata-Flavor': 'Google'})
    return response.content.decode('utf-8')

  def __init__(self, tpus, vms=None, zone=None, project=None):
    """Creates a new ClusterResolver object."""

    if not isinstance(tpus, list) or len(tpus) == 0:
      raise ValueError('tpus must be a non-empty list')
    if vms is not None:
      if not isinstance(vms, list) or len(vms) == 0:
        raise ValueError('vms must be a non-empty list if provided')

    self._tpus = tpus
    self._vms = vms
    self._zone = zone
    self._project = project

    self._credentials = GoogleCredentials.get_application_default()
    self._tpu_service = discovery.build(
        'tpu', 'v1', credentials=self._credentials, cache_discovery=False)
    self._compute_service = discovery.build(
        'compute', 'v1', credentials=self._credentials, cache_discovery=False)

    if project is None:
      self._project = self._get_instance_metadata('project/project-id')
    if zone is None:
      zone_path = self._get_instance_metadata('instance/zone')
      self._zone = zone_path.split('/')[-1]
    self._vm_master = self._get_instance_metadata('instance/name')

  def _get_instance_group(self):
    """Gets the instance group that the current VM belongs to."""
    resp = self._compute_service.instances().get(
        project=self._project,
        zone=self._zone,
        instance=self._vm_master,
        fields='metadata').execute()

    if 'metadata' in resp and 'items' in resp['metadata']:
      for item in resp['metadata']['items']:
        if (item['key'] == 'created-by' and
            'instanceGroupManagers' in item['value']):
          return item['value'].split('/')[-1]

    raise RuntimeError(('A vm list must be passed to ClusterResolver '
                        'if not using an instance group'))

  def _get_member_instance_names(self, instance_group):
    """Gets all the instance names that belong to the given instance group."""
    resp = self._compute_service.instanceGroups().listInstances(
        project=self._project, zone=self._zone,
        instanceGroup=instance_group).execute()

    instances = []
    if 'items' in resp:
      for item in resp['items']:
        if 'instance' not in item or 'status' not in item:
          continue
        instance_path = item['instance']
        instances.append(instance_path.split('/')[-1])

    return instances

  def _get_client_workers(self):
    """Gets client workers.

    The instance group that the current VM belongs to is picked up from
    the GCE instance metadata set of the VM. If a list of VMs was used for
    initializing cluster resolver, we use that instead.

    Returns:
      A list of ClientWorker.

    Raises:
      RuntimeError: If the red VM cluster is not healthy.
    """
    if self._vms is None:
      # Using an instance group
      instance_group = self._get_instance_group()
      self._vms = self._get_member_instance_names(instance_group)

    workers = []
    batch = BatchHttpRequest()
    
    def add_worker(request_id, resp, exception):
      """Callback for each request in BatchHttpRequest."""
      if exception is not None:
        raise exception
      hostname = resp['selfLink'].split('/')[-1]
      if resp['status'] != 'RUNNING':
        raise RuntimeError(
            ('Instance {hostname} is not running yet. '
             'Re-run when all VMs are running.').format(hostname=hostname))
      worker = ClientWorker(
          internal_ip=resp['networkInterfaces'][0]['networkIP'],
          machine_type=resp['machineType'].split('/')[-1],
          zone=resp['zone'].split('/')[-1],
          hostname=hostname)
      workers.append(worker)

    for vm in self._vms:
      req = self._compute_service.instances().get(
          project=self._project,
          zone=self._zone,
          instance=vm,
          fields=('machineType,metadata,selfLink,'
                  'networkInterfaces/networkIP,status,zone'))
      batch.add(req, add_worker)
    batch.execute()

    return workers

  def _get_service_workers(self):
    """Gets TPU VM cluster info.

    Calls the TPU CLH to get TPU node data and returns list of TPU worker
    VMs internal IP addresses. If zone and project are not specified at
    ClusterResolver init time, we infer these bits from GCE metadata.

    Returns:
      A list of ServiceWorker.

    Raises:
      RuntimeError: If the TPU DNE or the TPU is in not in HEALTHY state.
    """
    raise NotImplementedError()

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

