#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import re
import requests
import subprocess
import sys
import threading

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


class Worker(object):

  def __init__(self, internal_ip, machine_type, zone):
    if not isinstance(internal_ip, str):
      raise ValueError('internal_ip must be of type str')
    self._internal_ip = internal_ip
    if not isinstance(machine_type, str):
      raise ValueError('machine_type must be of type str')
    self._machine_type = machine_type
    if not isinstance(zone, str):
      raise ValueError('zone must be of type str')
    self._zone = zone


class ClientWorker(Worker):

  def __init__(self, internal_ip, machine_type, zone, hostname=None):
    super(ClientWorker, self).__init__(internal_ip, machine_type, zone)
    if hostname is not None and not isinstance(hostname, str):
      raise ValueError('hostname must be of type str')
    self._hostname = hostname

  def __repr__(self):
    return ('{{{internal_ip}, {machine_type}, {zone},'
            ' {hostname}}}').format(
                internal_ip=self._internal_ip,
                machine_type=self._machine_type,
                zone=self._zone,
                hostname=self._hostname)

  def __eq__(self, other):
    return (self._internal_ip == other._internal_ip and
            self._machine_type == other._machine_type and
            self._zone == other._zone and self._hostname == other._hostname)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(repr(self))

  def __repr__(self):
    return ('{{{internal_ip}, {machine_type}, {zone},'
            ' {hostname}}}').format(
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

  def __init__(self, internal_ip, port, machine_type, zone, sw_version):
    super(ServiceWorker, self).__init__(internal_ip, machine_type, zone)
    self._port = int(port)
    if not isinstance(sw_version, str):
      raise ValueError('sw_version must be of type str')
    self._sw_version = sw_version

  def __repr__(self):
    return ('{{{internal_ip}, {port}, {machine_type}, {zone},'
            ' {sw_version}}}').format(
                internal_ip=self._internal_ip,
                port=self._port,
                machine_type=self._machine_type,
                zone=self._zone,
                sw_version=self._sw_version)

  def __eq__(self, other):
    return (self._internal_ip == other._internal_ip and
            self._port == other._port and
            self._machine_type == other._machine_type and
            self._zone == other._zone and self._sw_version == other._sw_version)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(repr(self))


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

    sw_versions = {worker._sw_version for worker in self._service_workers}
    if len(sw_versions) != 1:
      raise RuntimeError(
          'All service workers must have the same sw_version, got: {}'.format(
              zones))

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


class ClusterResolver(object):
  """Cluster Resolver for Client VM and Cloud TPU mesh."""

  @staticmethod
  def _get_instance_metadata(metadata):
    response = requests.get(
        '{}/computeMetadata/v1/{}'.format(_GCE_METADATA_ENDPOINT, metadata),
        headers={'Metadata-Flavor': 'Google'})
    return response.content.decode('utf-8')

  @staticmethod
  def _parse_resource_url(url, name):
    parts = url.split('/')
    idx = parts.index(name)
    return parts[idx + 1]

  def __init__(self, tpus, vms=None, zone=None, project=None):
    """Creates a new ClusterResolver object."""

    if not isinstance(tpus, list) or len(tpus) == 0:
      raise ValueError('tpus must be a non-empty list')
    if vms:
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
      self._zone = self._parse_resource_url(zone_path, 'zones')
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
          return self._parse_resource_url(item['value'],
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
      instances.append(self._parse_resource_url(instance_path, 'instances'))

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
      hostname = self._parse_resource_url(resp['selfLink'], 'instances')
      if resp['status'] != 'RUNNING':
        raise RuntimeError(
            ('Instance {hostname} is not running yet. '
             'Re-run when all VMs are running').format(hostname=hostname))
      worker = ClientWorker(
          internal_ip=resp['networkInterfaces'][0]['networkIP'],
          machine_type=self._parse_resource_url(resp['machineType'],
                                                'machineTypes'),
          zone=self._parse_resource_url(resp['zone'], 'zones'),
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

  def get_service_workers(self):
    """Gets TPU VM cluster info.

    Calls the TPU CLH to get TPU node data and returns list of TPU worker
    VMs internal IP addresses. If zone and project are not specified at
    ClusterResolver init time, we infer these bits from GCE metadata.

    Returns:
      A list of ServiceWorker.

    Raises:
      RuntimeError: If the TPU DNE or the TPU is in not in HEALTHY state.
    """
    workers = []
    batch = self._tpu_service.new_batch_http_request()

    def add_service_worker(request_id, resp, exception):
      """Callback for each request in BatchHttpRequest."""
      if exception is not None:
        raise exception
      tpu_name = self._parse_resource_url(resp['name'], 'nodes')
      zone = self._parse_resource_url(resp['name'], 'locations')
      if resp['state'] != 'READY':
        raise RuntimeError(
            ('TPU {tpu_name} is not READY yet. '
             'Re-run when all TPUs are READY').format(tpu_name=tpu_name))
      if 'health' not in resp or resp['health'] != 'HEALTHY':
        raise RuntimeError(
            ('TPU {tpu_name} is not HEALTHY yet. '
             'Re-run when all TPUs are HEALTHY').format(tpu_name=tpu_name))
      sw_version = resp['tensorflowVersion']
      machine_type = resp['acceleratorType']
      for endpoint in resp['networkEndpoints']:
        worker = ServiceWorker(
            internal_ip=endpoint['ipAddress'],
            port=endpoint['port'],
            machine_type=machine_type,
            zone=zone,
            sw_version=sw_version)
        workers.append(worker)

    for tpu in self._tpus:
      tpu_fq_name = 'projects/{project}/locations/{location}/nodes/{node}'.format(
          project=self._project, location=self._zone, node=tpu)
      req = self._tpu_service.projects().locations().nodes().get(
          name=tpu_fq_name,
          fields=('acceleratorType,health,ipAddress,name,'
                  'networkEndpoints,state,tensorflowVersion'))
      batch.add(req, add_service_worker)
    batch.execute()

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
    client_workers = self.get_client_workers()
    service_workers = self.get_service_workers()
    cluster = Cluster(client_workers, service_workers)
    cluster.validate()
    return cluster


class DistributedExecutor(object):

  SCRIPT_PATH_TMPL = '/tmp/{pid}/dist_training_ptxla_{worker}.sh'
  MASTER_IDX = 0
  MESH_SERVICE_PORT = 8477  # Use single port to disallow concurrent runs
  DIST_ENV_VARS = [
      'XRT_TPU_CONFIG', 'XRT_LOCAL_WORKER', 'XRT_MESH_SERVICE_ADDRESS'
  ]
  DEFAULT_CONTAINER_NAME = 'pytorchtpudistrunner'
  DEFAULT_USER_NAME = 'pytorchtpudistrunner'

  def __init__(self,
               cluster,
               docker_container=None,
               docker_image=None,
               docker_run_flags=None,
               conda_env=None,
               env_vars=None):
    self._cluster = cluster
    logging.basicConfig(
        format='%(asctime)-12s %(clientip)s [%(ordinal)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)
    self.logger = logging.getLogger('DistributedExecutor')
    self.docker_container = docker_container or self.DEFAULT_CONTAINER_NAME
    self.docker_image = docker_image
    self.docker_run_flags = docker_run_flags
    self.conda_env = conda_env
    self.env_vars = env_vars

    for env_var in env_vars:
      if re.match('\w*=\w*', env_var) is None:
        raise ValueError(
            ('Environment variable to distribute ({}) should follow '
             'the form: X=Y').format(env_var))
      for dist_var in self.DIST_ENV_VARS:
        if re.match('{}=.*'.format(dist_var), env_var):
          raise ValueError(
              ('{} should not be in the training command provided as they'
               ' will interfere with the values set for distributed'
               ' training'.format(dist_var)))

  def _stream_logs(self, process, client_worker):
    client_ip = client_worker._internal_ip
    ordinal = self._cluster._client_workers.index(client_worker)

    def _stream_output(stream, log_fn):
      for std in iter(stream.readline, b''):
        log_fn(
            std.decode('utf-8').rstrip('\n'),
            extra={
                'clientip': client_ip,
                'ordinal': ordinal
            })

    stdout = threading.Thread(
        target=_stream_output, args=(
            process.stdout,
            self.logger.info,
        ))
    stdout.daemon = True
    stdout.start()
    stderr = threading.Thread(
        target=_stream_output, args=(
            process.stderr,
            self.logger.error,
        ))
    stderr.daemon = True
    stderr.start()
    stdout.join()
    stderr.join()

  def _build_scp_cmd(self, local_path, remote_path, client_worker):
    return [
        'gcloud',
        '-q',
        'compute',
        'scp',
        '--internal-ip',
        '--zone={}'.format(client_worker._zone),
        local_path,
        '{}@{}:~/{}'.format(self.DEFAULT_USER_NAME, client_worker._hostname,
                            os.path.basename(remote_path)),
    ]

  def _build_ssh_cmd(self, remote_cmd, client_worker):
    if isinstance(remote_cmd, list):
      remote_cmd = ' '.join('"{}"'.format(c) for c in remote_cmd)
    return [
        'gcloud',
        '-q',
        'compute',
        'ssh',
        '--internal-ip',
        '--zone={}'.format(client_worker._zone),
        '{}@{}'.format(self.DEFAULT_USER_NAME, client_worker._hostname),
        '--command="{}"'.format(remote_cmd),
    ]

  def _run_remote_cmd(self, cmd, client_worker, shell=True, log=True):
    cmd = ' '.join(cmd) if shell else cmd
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
    if log:
      self._stream_logs(proc, client_worker)
    proc.wait()

  def _build_and_run_ssh(self, remote_cmd, client_worker, shell=True, log=True):
    cmd = self._build_ssh_cmd(remote_cmd, client_worker)
    self._run_remote_cmd(cmd, client_worker, shell=shell, log=log)

  def _docker_run_cmd(self, cmd):
    docker_cmd = [
        'docker',
        'run',
        '--name={}'.format(self.docker_container),
        '--network=host',
    ]
    docker_cmd.extend(self.docker_run_flags)
    for env_var in self.DIST_ENV_VARS:
      docker_cmd.extend(['-e', env_var])
    for env_kv in self.env_vars:
      key = re.match('(\w*)=.*', env_kv)
      if key:
        docker_cmd.extend(['-e', key.group(1)])
    docker_cmd.append(self.docker_image)
    docker_cmd.extend(cmd)
    return docker_cmd

  def _env_vars_cmd(self, worker_idx):
    client_master = self._cluster._client_workers[self.MASTER_IDX]
    env_vars = {
        'XRT_LOCAL_WORKER':
            'c_tpu_worker:{}'.format(worker_idx),
        'XRT_MESH_SERVICE_ADDRESS':
            '{}:{}'.format(client_master._internal_ip, self.MESH_SERVICE_PORT)
    }
    # Only for master
    if worker_idx == self.MASTER_IDX:
      xrt_server_config = [
          'c_tpu_worker;{worker_idx};{worker_ip}:{worker_port}'.format(
              worker_idx=idx,
              worker_ip=service_worker._internal_ip,
              worker_port=service_worker._port)
          for idx, service_worker in enumerate(self._cluster._service_workers)
      ]
      xrt_tpu_config = '|'.join(xrt_server_config)
      env_vars['XRT_TPU_CONFIG'] = '"{}"'.format(xrt_tpu_config)

    export_cmd = []
    for k in env_vars:
      export_cmd.append(['export', '{}={}'.format(k, env_vars[k])])
    for kv in self.env_vars:
      export_cmd.append(['export', '{}'.format(kv)])
    return export_cmd

  def _prepare_scripts(self, cmd):
    worker_script_map = {}
    for i in range(len(self._cluster._client_workers)):
      script_path = self.SCRIPT_PATH_TMPL.format(pid=os.getpid(), worker=i)

      # ex. script = [['conda', 'activate', 'pytorch'], ['python', 'train.py']]
      script = []
      script.extend(self._env_vars_cmd(i))
      # Setup environment for non-interactive non-login shell over ssh
      script.append(['.', '/etc/profile'])
      if self.docker_image:
        script.append(self._docker_run_cmd(cmd))
      else:
        if self.conda_env:
          script.append(['conda', 'activate', self.conda_env])
        script.append(cmd)

      # ex. script_body = 'conda activate pytorch; python train.py'
      script_body = '; '.join([' '.join(command) for command in script])
      os.makedirs(os.path.dirname(script_path), exist_ok=True)
      with open(script_path, 'w') as f:
        f.write(script_body)
      subprocess.call(['chmod', '+x', script_path])
      worker_script_map[self._cluster._client_workers[i]] = script_path

    return worker_script_map

  def _scp_scripts(self, script_map):

    def _gcloud_scp(script_path, client_worker):
      scp_cmd = self._build_scp_cmd(script_path, os.path.basename(script_path),
                                    client_worker)
      proc = subprocess.Popen(
          scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      self._stream_logs(proc, client_worker)

    threads = []
    for i, client_worker in enumerate(script_map):
      if i == 0:
        # ssh keygen single time
        _gcloud_scp(script_map[client_worker], client_worker)
        continue
      thread = threading.Thread(
          target=_gcloud_scp, args=(
              script_map[client_worker],
              client_worker,
          ))
      thread.daemon = True
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def _cleanup(self, script_map):

    def _cleanup_worker(script_path, client_worker):
      rm_script = ['rm', '~/{}'.format(os.path.basename(script_path))]
      self._build_and_run_ssh(rm_script, client_worker)
      subprocess.call(['rm', script_path])
      if self.docker_image:
        rm_container = ['docker', 'rm', '-f', self.docker_container]
        self._build_and_run_ssh(rm_container, client_worker)
      rm_proc = ['pkill', '-u', self.DEFAULT_USER_NAME]
      self._build_and_run_ssh(rm_proc, client_worker, log=False)

    threads = []
    for client_worker in script_map:
      thread = threading.Thread(
          target=_cleanup_worker,
          args=(
              script_map[client_worker],
              client_worker,
          ))
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

  def _start_run(self, script_map):

    def _run_script(script_path, client_worker):
      self._build_and_run_ssh(
          ['~/{}'.format(os.path.basename(script_path))], client_worker)

    threads = []
    for client_worker in script_map:
      thread = threading.Thread(
          target=_run_script, args=(
              script_map[client_worker],
              client_worker,
          ))
      thread.daemon = True
      thread.start()
      threads.append(thread)

    try:
      for thread in threads:
        thread.join()
    except KeyboardInterrupt:
      pass
    finally:
      self._cleanup(script_map)

    for thread in threads:
      thread.join()

  def run(self, cmd):
    self.logger.info(
        'Command to distribute: {}'.format(' '.join(cmd)),
        extra={
            'clientip': '',
            'ordinal': ''
        })
    self.logger.info(
        'Cluster configuration: {}'.format(self._cluster),
        extra={
            'clientip': '',
            'ordinal': ''
        })
    script_map = self._prepare_scripts(cmd)
    self._scp_scripts(script_map)
    self._start_run(script_map)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='PyTorch on TPU distrubuted training',
      epilog=('Usage example: xla_dist.py --tpus=[TPU_NAME]'
              ' --conda-env pytorch-nightly -- python train'))

  cluster_group = parser.add_argument_group('Cluster Setup')
  cluster_group.add_argument(
      '--tpus',
      nargs='+',
      type=str,
      help='Name of the TPU pod, or list of single Cloud TPU devices (v*-8).')
  cluster_group.add_argument(
      '--vms',
      nargs='+',
      default='',
      type=str,
      required=False,
      help=('List of single Compute VM instance names. '
            'If not provided we assume usage of instance groups.'))

  docker_group = parser.add_argument_group('Docker Setup')
  docker_group.add_argument(
      '--docker-container',
      default='',
      type=str,
      required=False,
      help='Name of docker container if running in docker.')
  docker_group.add_argument(
      '--docker-image',
      default='',
      type=str,
      required=False,
      help='Name of docker image if running in container.')
  docker_group.add_argument(
      '--docker-run-flag',
      nargs='+',
      default='',
      type=str,
      help='Docker run flags to run container with (ex. --shm-size, ...).')

  conda_group = parser.add_argument_group('Conda Setup')
  conda_group.add_argument(
      '--conda-env',
      default='',
      type=str,
      required=False,
      help='Name of the conda environment if running with conda.')

  parser.add_argument(
      '--env',
      nargs='+',
      type=str,
      help='List of environment variables to distribute.')
  parser.add_argument(
      'positional',
      nargs='+',
      default='',
      type=str,
      help='The python command to launch training including model parameters.')

  FLAGS = parser.parse_args()

  if (FLAGS.docker_container or FLAGS.docker_image or
      FLAGS.docker_run_flag) and FLAGS.conda_env:
    raise ValueError('Docker Setup arguments and Conda Setup'
                     ' arguments are mutually exclusive.')

  # Resolve VM and TPU clusters.
  cluster_resolver = ClusterResolver(FLAGS.tpus, vms=FLAGS.vms)
  cluster = cluster_resolver.get_cluster()
  executor = DistributedExecutor(
      cluster,
      docker_container=FLAGS.docker_container,
      docker_image=FLAGS.docker_image,
      docker_run_flags=FLAGS.docker_run_flag,
      conda_env=FLAGS.conda_env,
      env_vars=FLAGS.env)
  executor.run(FLAGS.positional)
