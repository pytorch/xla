import cloud_tpu_client
import logging
import multiprocessing
import re
import requests
import subprocess
import time

from torch_xla.distributed.worker import ClientWorker
from torch_xla.distributed.worker import ServiceWorker
import torch_xla.utils.utils as xu

try:
  from googleapiclient import discovery
  from oauth2client.client import GoogleCredentials
except ImportError:
  raise ImportError('googleapiclient and oauth2client must be installed '
                    'before using the xla_dist. Execute: '
                    '`pip install --upgrade google-api-python-client` '
                    'and `pip install --upgrade oauth2client` to '
                    'install with pip')

_GCE_METADATA_ENDPOINT = 'http://metadata.google.internal'

# Silence noisy loggging
logging.getLogger('oauth2client').setLevel(logging.ERROR)
logging.getLogger('googleapiclient').setLevel(logging.ERROR)


class Cluster(object):

  def __init__(self,
               client_workers,
               service_workers,
               check_client_machine_type=True,
               check_service_machine_type=True,
               client_master_ip=None):
    """Creates a cluster object.

    Args:
      client_workers: a list of ClientWorker objects.
      service_workers: a list of ServiceWorker objects.
      check_client_machine_type: whether to check if client workers all have the
        same machine type.
      check_service_machine_type: whether to check if service workers all have
        the same machine type.
      client_master_ip: the ip of client worker to set as master. If not
        provided, the VM running the current process is the master.
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

    if not client_master_ip:
      client_master_ip = ClusterResolver.get_instance_metadata(
          'instance/network-interfaces/0/ip')
    self._client_master = next(
        filter(lambda cw: cw.get_internal_ip() == client_master_ip,
               self._client_workers))

    # Put client master at front of client worker list.
    self._client_workers.remove(self._client_master)
    self._client_workers.insert(0, self._client_master)

  def get_client_master(self):
    return self._client_master

  def get_client_workers(self):
    return self._client_workers

  def get_service_workers(self):
    return self._service_workers

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
            'All client_workers must have the same machine_type, got: {}'.
            format(client_machine_types))

    if self._check_service_machine_type:
      server_machine_types = {
          worker._machine_type for worker in self._service_workers
      }
      if len(server_machine_types) != 1:
        raise RuntimeError(
            'All service_workers must have the same machine_type, got: {}'.
            format(server_machine_types))

    runtime_versions = {
        worker._runtime_version for worker in self._service_workers
    }
    if len(runtime_versions) != 1:
      raise RuntimeError(
          'All service workers must have the same runtime_version, got: {}'.
          format(zones))

  def __eq__(self, other):
    return (self._client_workers == other._client_workers and
            self._service_workers == other._service_workers)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    return ('{{client_workers: {client_workers}, '
            'service_workers: {service_workers}}}').format(
                client_workers=self._client_workers,
                service_workers=self._service_workers)

  def list_tpus_with_health(self, health):

    def _tpu_with_health(tpu_name):
      ctc = cloud_tpu_client.Client(tpu_name)
      if ctc.health() == health:
        return tpu_name

    tpus = set()
    for service_worker in self._service_workers:
      tpus.add(service_worker._tpu)
    results = xu.parallel_work(len(tpus), _tpu_with_health, tpus)
    return [res for res in results if res]

  def wait_for_healthy_service(self):

    def wait_for_healthy_service_worker(tpu_name):
      ctc = cloud_tpu_client.Client(tpu=tpu_name)
      ctc.wait_for_healthy()

    tpus = self.list_tpus_with_health('UNHEALTHY_MAINTENANCE')
    if tpus:
      xu.parallel_work(len(tpus), wait_for_healthy_service_worker, tpus)

  def wait_for_healthy_client(self, dist_executor, timeout=1200, interval=10):

    def wait_for_healthy_client_worker(client_worker):
      heartbeart_check = [
          'echo', 'client_worker', '$(hostname)', 'is', 'healthy'
      ]
      check_timeout = time.time() + timeout

      def _healthy_client_worker():
        proc = multiprocessing.Process(
            target=dist_executor._build_and_run_ssh,
            args=(
                heartbeart_check,
                client_worker,
            ))
        proc.daemon = True
        proc.start()
        proc.join(interval)

        if proc.is_alive():
          proc.terminate()
          return False

        return proc.exitcode == 0

      while not _healthy_client_worker():
        logging.warning(
            'Waiting for client_worker "{}" to become healthy'.format(
                client_worker))
        if time.time() + interval > check_timeout:
          raise RuntimeError(
              'Timed out waiting for client_worker {} to become healthy'.format(
                  client_worker))

      logging.warning('client_worker "{}" is healthy.'.format(client_worker))

    xu.parallel_work(
        len(self._client_workers), wait_for_healthy_client_worker,
        self._client_workers)


class ClusterResolver(object):
  """Cluster Resolver for Client VM and Cloud TPU mesh."""

  @staticmethod
  def get_instance_metadata(metadata):
    response = requests.get(
        '{}/computeMetadata/v1/{}'.format(_GCE_METADATA_ENDPOINT, metadata),
        headers={'Metadata-Flavor': 'Google'})
    return response.content.decode('utf-8')

  @staticmethod
  def _parse_resource_url(url, name):
    parts = url.split('/')
    idx = parts.index(name)
    return parts[idx + 1]

  @staticmethod
  def _get_internal_ip_to_hostname_mapping(tpu_name, zone, num_vm):
    """Gets TPU VM internal IP to hostname mapping.

    Currently TPU CLH does not expose any TPU host machine name. SSH to each worker and
    get that instead.

    Returns:
      A map of TPU VM internal IP to TPU VM hostname.
    """
    ip_to_host_name = {}

    def add_tpuvm_ip_to_hostname_mapping(worker_index):
      proc = subprocess.Popen([
          'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh',
          '--internal-ip', tpu_name, '--zone', zone, '--worker',
          str(worker_index), '--command', 'hostname; hostname -i'
      ],
                              stdout=subprocess.PIPE)
      hostname = proc.stdout.readline().decode('utf-8').rstrip('\n')
      ip = proc.stdout.readline().decode('utf-8').rstrip('\n')
      ip_to_host_name[ip] = hostname

    xu.parallel_work(num_vm, add_tpuvm_ip_to_hostname_mapping,
                     list(range(num_vm)))
    return ip_to_host_name

  def __init__(self, tpu, vms=None, zone=None, project=None):
    """Creates a new ClusterResolver object."""

    if not tpu:
      raise ValueError('tpu must be a non-empty string')
    if vms:
      if not isinstance(vms, list) or len(vms) == 0:
        raise ValueError('vms must be a non-empty list if provided')

    self._tpus = tpu if isinstance(tpu, list) else [tpu]
    self._vms = vms
    self._zone = zone
    self._project = project
    self._tpuvm_mode = None
    self._tpuvm_mode_with_remote_coordinator = None
    self._set_tpuvm_mode()

    self._compute_service = discovery.build(
        'compute',
        'v1',
        credentials=GoogleCredentials.get_application_default(),
        cache_discovery=False)

    if project is None:
      self._project = ClusterResolver.get_instance_metadata(
          'project/project-id')
    if zone is None:
      zone_path = ClusterResolver.get_instance_metadata('instance/zone')
      self._zone = ClusterResolver._parse_resource_url(zone_path, 'zones')
    self._vm_master = ClusterResolver.get_instance_metadata('instance/name')

  def _set_tpuvm_mode(self):
    self._tpuvm_mode = False
    self._tpuvm_mode_with_remote_coordinator = False
    accel_type = ClusterResolver.get_instance_metadata(
        'instance/attributes/accelerator-type')
    if re.match(r'v[0-9]+-[0-9]+', accel_type):
      # Only VM with TPU attched will carry the accelerator-type metadata
      self._tpuvm_mode = True
      return

    api_version = cloud_tpu_client.Client(
        tpu=self._tpus[0])._get_tpu_property('apiVersion')
    if api_version == 'V2_ALPHA1':
      # Only TPUVM api version should be V2_ALPHA1
      self._tpuvm_mode = True
      # Current vm does not carry the accelerator-type metadata but tpu specified
      # is a TPUVM, assume it is a remote coordinator.
      self._tpuvm_mode_with_remote_coordinator = True

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
          return ClusterResolver._parse_resource_url(item['value'],
                                                     'instanceGroupManagers')

    raise RuntimeError(('A vm list must be passed to ClusterResolver '
                        'if not using an instance group'))

  def _get_member_instance_names(self, instance_group):
    """Gets all the instance names that belong to the given instance group."""
    resp = self._compute_service.instanceGroups().listInstances(
        project=self._project, zone=self._zone,
        instanceGroup=instance_group).execute()

    instances = []
    for item in resp.get('items', []):
      if 'instance' not in item or 'status' not in item:
        continue
      instance_path = item['instance']
      instances.append(
          ClusterResolver._parse_resource_url(instance_path, 'instances'))

    return instances

  def get_client_workers(self):
    """Gets client workers.

    The instance group that the current VM belongs to is picked up from
    the GCE instance metadata set of the VM. If a list of VMs was used for
    initializing cluster resolver, we use that instead.

    Returns:
      A list of ClientWorker.

    Raises:
      RuntimeError: If the red VM cluster is not healthy.
    """
    if not self._vms:
      # Using an instance group
      instance_group = self._get_instance_group()
      self._vms = self._get_member_instance_names(instance_group)
      if len(self._vms) == 0:
        raise RuntimeError('Client worker vms is empty in instance group')

    workers = []
    batch = self._compute_service.new_batch_http_request()

    def add_client_worker(request_id, resp, exception):
      """Callback for each request in BatchHttpRequest."""
      if exception is not None:
        raise exception
      hostname = ClusterResolver._parse_resource_url(resp['selfLink'],
                                                     'instances')
      if resp['status'] != 'RUNNING':
        raise RuntimeError(
            ('Instance {hostname} is not running yet. '
             'Re-run when all VMs are running').format(hostname=hostname))
      worker = ClientWorker(
          internal_ip=resp['networkInterfaces'][0]['networkIP'],
          machine_type=ClusterResolver._parse_resource_url(
              resp['machineType'], 'machineTypes'),
          zone=ClusterResolver._parse_resource_url(resp['zone'], 'zones'),
          hostname=hostname)
      workers.append(worker)

    for vm in self._vms:
      req = self._compute_service.instances().get(
          project=self._project,
          zone=self._zone,
          instance=vm,
          fields=('machineType,metadata,selfLink,'
                  'networkInterfaces/networkIP,status,zone'))
      batch.add(req, add_client_worker)
    batch.execute()

    return workers

  def get_tpu_workers(self, as_client_worker=False):
    """Gets TPU VM cluster info.

    Calls the TPU CLH to get TPU node data and returns list of TPU worker
    VMs internal IP addresses. If zone and project are not specified at
    ClusterResolver init time, we infer these bits from GCE metadata.

    Returns:
      A list of ServiceWorker or a list of ClientWorker.

    Raises:
      RuntimeError: If the TPU DNE or the TPU is in not in HEALTHY state.
    """
    workers = []

    def add_tpu_worker(tpu_name):
      ctc = cloud_tpu_client.Client(tpu=tpu_name)
      tpu_name = ctc.name()
      if ctc.state() != 'READY':
        raise RuntimeError(
            ('TPU {tpu_name} is not READY yet. '
             'Re-run when all TPUs are READY').format(tpu_name=tpu_name))
      if ctc.health() != 'HEALTHY':
        raise RuntimeError(
            ('TPU {tpu_name} is not HEALTHY yet. '
             'Re-run when all TPUs are HEALTHY').format(tpu_name=tpu_name))

      runtime_version = ctc.runtime_version()
      machine_type = ctc.accelerator_type()
      zone = ClusterResolver._parse_resource_url(ctc._full_name(), 'locations')
      network_endpoints = ctc.network_endpoints()

      if as_client_worker:
        ip_to_host_name = ClusterResolver._get_internal_ip_to_hostname_mapping(
            tpu_name, zone, len(network_endpoints))

      for endpoint in network_endpoints:
        if as_client_worker:
          internal_ip = endpoint['ipAddress']
          hostname = ip_to_host_name[internal_ip]
          worker = ClientWorker(
              internal_ip=internal_ip,
              machine_type=machine_type,
              zone=zone,
              hostname=hostname)
        else:
          worker = ServiceWorker(
              internal_ip=endpoint['ipAddress'],
              port=endpoint['port'],
              machine_type=machine_type,
              zone=zone,
              runtime_version=runtime_version,
              tpu=tpu_name)
        workers.append(worker)

    xu.parallel_work(len(self._tpus), add_tpu_worker, self._tpus)

    return workers

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
    service_workers = self.get_tpu_workers(as_client_worker=False)
    client_workers = self.get_tpu_workers(
        as_client_worker=True) if self._tpuvm_mode else self.get_client_workers(
        )
    client_master_ip = None
    if self._tpuvm_mode_with_remote_coordinator:
      # If the script is being run from a remote coordinator with a TPUVM, client_master_ip
      # should be TPUVM IP instead of the remote coordinator IP.
      client_master_ip = client_workers[0].get_internal_ip()
    cluster = Cluster(
        client_workers, service_workers, client_master_ip=client_master_ip)
    cluster.validate()
    return cluster

  def get_tpuvm_mode(self):
    return self._tpuvm_mode
