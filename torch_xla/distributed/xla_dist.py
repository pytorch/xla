#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function

import argparse
import cloud_tpu_client
import logging
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import time
import threading
import torch_xla.core.xla_env_vars as xenv
from torch_xla.distributed.cluster import ClusterResolver
import torch_xla.utils.utils as xu


def concat_cmd_list(cmd_list, delimiter=' ', quote='"'):
  concat = ''
  for cmd in cmd_list:
    if re.match('^{}.*{}$'.format(quote, quote), cmd):
      token = cmd
    else:
      token = quote + cmd + quote
    if concat:
      concat += delimiter
    concat += token
  return concat


class DistributedExecutor(object):

  SCRIPT_PATH_TMPL = '/tmp/{pid}/dist_training_ptxla_{worker}.sh'
  XRT_RUN_SERVER_CMD = 'torch_xla.core.xrt_run_server'
  XRT_RUN_SERVER_PROCESS = 'torch_xla.core._xrt_run_server'
  MESH_SERVICE_PORT = 8477  # Use single port to disallow concurrent runs
  DIST_ENV_VARS = [
      xenv.TPU_CONFIG,
      xenv.LOCAL_WORKER,
      xenv.SERVICE_ADDRESS,
      xenv.WORLD_SIZE,
      xenv.ORDINAL,
      xenv.TPU_NUM_DEVICES,
      'XLA_EMIT_STEPLOG',
  ]
  DEFAULT_CONTAINER_NAME = 'pytorchtpudistrunner'
  MAX_TPU_RETRY = 50
  HEARTBEAT_CHECK_PERIOD = 30

  def _get_logger(self):
    logger = logging.getLogger(self.__class__.__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt='%(asctime)-12s %(clientip)s [%(ordinal)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

  def _initialize(self):
    """Initializes members that need to be cleanly initialized for each run."""
    self._last_heartbeats = {
        cw.get_internal_ip(): {
            'last_time': time.time(),
            'count': 0,
        } for cw in self._cluster.get_client_workers()
    }
    self._error_queue = multiprocessing.Queue()
    self._last_heartbeat_check_time = 0

  def __init__(self,
               cluster,
               docker_container=None,
               docker_image=None,
               docker_run_flags=None,
               conda_env=None,
               env_vars=None,
               restart_server=None,
               tpuvm_mode=None,
               tpuvm_server_port=None):
    self._cluster = cluster
    self._initialize()
    self.logger = self._get_logger()
    self.docker_container = docker_container or self.DEFAULT_CONTAINER_NAME
    self.docker_image = docker_image
    self.docker_run_flags = list(docker_run_flags) if docker_run_flags else []
    self.conda_env = conda_env
    self.env_vars = list(env_vars) if env_vars else []
    self.tpuvm_mode = tpuvm_mode
    self.restart_server = restart_server
    self.tpuvm_server_port = tpuvm_server_port
    self.tpu_name = self._cluster.get_service_workers()[0]._tpu

    for env_var in self.env_vars:
      if re.match(r'\w*=\w*', env_var) is None:
        raise ValueError(
            ('Environment variable to distribute ({}) should follow '
             'the form: X=Y').format(env_var))
      for dist_var in self.DIST_ENV_VARS:
        if re.match('{}=.*'.format(dist_var), env_var):
          raise ValueError(
              ('{} should not be in the training command provided as they'
               ' will interfere with the values set for distributed'
               ' training'.format(dist_var)))

  def _check_client_mesh_health(self, uneven_health_timeout,
                                even_health_timeout):
    min_delay = max(uneven_health_timeout, even_health_timeout) + 1
    count = None
    now = time.time()
    if xu.getenv_as('XLA_DEBUG_LOG_HEARTBEATS', bool, False):
      self.logger.info(
          'Worker Heartbeats: {}'.format(self._last_heartbeats),
          extra={
              'clientip': '',
              'ordinal': ''
          })

    for cw_hb in self._last_heartbeats.values():
      min_delay = min(min_delay, now - cw_hb['last_time'])
      if count is None:
        count = cw_hb['count']
      elif count >= 0 and count != cw_hb['count']:
        count = -1

    if count < 0 and min_delay > uneven_health_timeout:
      self._error_queue.put(
          RuntimeError('Client mesh is unhealthy with uneven heartbeats'))
    elif count > 0 and min_delay > even_health_timeout:
      self._error_queue.put(
          RuntimeError('Client mesh is unhealthy with even heartbeats'))

  def _stream_logs(self, process, client_worker):
    client_ip = client_worker.get_internal_ip()
    ordinal = self._cluster.get_client_workers().index(client_worker)

    def _stream_output(stream, log_fn):
      for std in iter(stream.readline, b''):
        std_line = std.decode('utf-8').rstrip('\n')
        if 'torch_xla.core.xla_model::mark_step' in std_line:
          hb_stream = self._last_heartbeats[client_ip]
          # Only single thread updates each of these, so there is no race
          hb_stream['last_time'] = time.time()
          hb_stream['count'] += 1
          continue
        log_fn(std_line, extra={'clientip': client_ip, 'ordinal': ordinal})

    stdout = threading.Thread(
        target=_stream_output,
        daemon=True,
        args=(
            process.stdout,
            self.logger.info,
        ))
    stdout.start()
    stderr = threading.Thread(
        target=_stream_output,
        daemon=True,
        args=(
            process.stderr,
            self.logger.error,
        ))
    stderr.start()
    stdout.join()
    stderr.join()

  def _is_retry(self):
    return self.trials >= 1

  def _build_scp_cmd(self, local_path, remote_path, client_worker):
    if not self._is_retry() and not self.tpuvm_mode:
      return [
          'gcloud',
          '-q',
          'compute',
          'scp',
          '--internal-ip',
          '--zone={}'.format(client_worker.get_zone()),
          local_path,
          '{}:{}'.format(client_worker.get_hostname(), remote_path),
      ]
    return [
        'scp',
        '-oStrictHostKeyChecking=no',
        '-i',
        '~/.ssh/google_compute_engine',
        local_path,
        '{}@{}:{}'.format(os.getlogin(), client_worker.get_hostname(),
                          remote_path),
    ]

  def _build_ssh_cmd(self, remote_cmd, client_worker):
    if isinstance(remote_cmd, list):
      remote_cmd = concat_cmd_list(remote_cmd)
    if not self._is_retry():
      if self.tpuvm_mode:
        return [
            'gcloud',
            'alpha',
            '-q',
            'compute',
            'tpus',
            'tpu-vm',
            'ssh',
            '--internal-ip',
            '{}'.format(self.tpu_name),
            '--zone {}'.format(client_worker.get_zone()),
            '--worker {}'.format(client_worker.get_hostname().split('-')[-1]),
            '--command',
            '\'{}\''.format(remote_cmd),
        ]
      else:
        return [
            'gcloud',
            '-q',
            'compute',
            'ssh',
            '--internal-ip',
            '--zone={}'.format(client_worker.get_zone()),
            '{}'.format(client_worker.get_hostname()),
            '--command',
            '\'{}\''.format(remote_cmd),
        ]
    return [
        'ssh',
        '-oStrictHostKeyChecking=no',
        '-i',
        '~/.ssh/google_compute_engine',
        '{}@{}'.format(os.getlogin(), client_worker.get_hostname()),
        '\'{}\''.format(remote_cmd),
    ]

  def _run_remote_cmd(self, cmd, client_worker, shell=True, log=True):
    cmd = concat_cmd_list(cmd, quote='') if shell else cmd
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
    if log:
      self._stream_logs(proc, client_worker)
    proc.wait()
    if proc.returncode == 255:
      self._error_queue.put(
          RuntimeError(
              'Client mesh is unhealthy due to dead client worker: {}'.format(
                  client_worker)))
    return proc.returncode

  def _build_and_run_ssh(self, remote_cmd, client_worker, shell=True, log=True):
    cmd = self._build_ssh_cmd(remote_cmd, client_worker)
    return self._run_remote_cmd(cmd, client_worker, shell=shell, log=log)

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
      key = re.match(r'(\w*)=.*', env_kv)
      if key:
        docker_cmd.extend(['-e', key.group(1)])
    docker_cmd.append(self.docker_image)
    docker_cmd.extend(cmd)
    return docker_cmd

  def _tpuvm_env_vars_cmd(self, worker_idx):
    env_vars = {
        xenv.TPU_CHIPS_PER_HOST_BOUNDS: '2,2,1',
        xenv.TPUVM_MODE: 1,
        xenv.CLOUD_TPU_TASK_ID: worker_idx,
    }
    accelerator_type = self._cluster.get_service_workers()[0]._machine_type
    master_worker_network_endpoints = self._cluster.get_client_workers(
    )[0].get_internal_ip()

    accelerator_type_to_host_bounds = {
        # v2
        'v2-8': '1,1,1',
        'v2-32': '2,2,1',
        'v2-128': '4,4,1',
        'v2-256': '4,8,1',
        'v2-512': '8,8,1',
        # v3
        'v3-8': '1,1,1',
        'v3-32': '2,2,1',
        'v3-64': '2,4,1',
        'v3-128': '4,4,1',
        'v3-256': '4,8,1',
        'v3-512': '8,8,1',
        'v3-1024': '8,16,1',
        'v3-2048': '16,16,1',
    }

    env_vars[xenv.TPU_HOST_BOUNDS] = accelerator_type_to_host_bounds[
        accelerator_type]
    env_vars[xenv.TPU_MESH_CTLER_ADDR] = '{}:{}'.format(
        master_worker_network_endpoints, '8476')
    env_vars[xenv.TPU_MESH_CTLER_PORT] = 8476
    return env_vars

  def _env_vars_cmd(self, worker_idx):
    client_worker = self._cluster.get_client_workers()[worker_idx]
    worker_name = 'c_localservice' if self.tpuvm_mode else 'c_tpu_worker'
    env_vars = {
        xenv.LOCAL_WORKER:
            '{}:{}'.format(worker_name, worker_idx),
        xenv.SERVICE_ADDRESS:
            '{}:{}'.format(self._cluster.get_client_master().get_internal_ip(),
                           self.MESH_SERVICE_PORT),
        xenv.WORLD_SIZE:
            len(self._cluster.get_client_workers()),
        xenv.ORDINAL:
            worker_idx,
        xenv.TPU_NUM_DEVICES:
            8,
        'XLA_EMIT_STEPLOG':
            1,
    }
    if self.tpuvm_mode:
      env_vars.update(self._tpuvm_env_vars_cmd(worker_idx))

    # Only for master
    if client_worker == self._cluster.get_client_master():
      xrt_server_config = [
          '{worker_name};{worker_idx};{worker_ip}:{worker_port}'.format(
              worker_name=worker_name,
              worker_idx=idx,
              worker_ip=service_worker.get_internal_ip(),
              worker_port=self.tpuvm_server_port
              if self.tpuvm_mode else service_worker.get_port()) for idx,
          service_worker in enumerate(self._cluster.get_service_workers())
      ]
      xrt_tpu_config = '|'.join(xrt_server_config)
      env_vars[xenv.TPU_CONFIG] = '{}'.format(xrt_tpu_config)

    export_cmd = []
    for k in env_vars:
      export_cmd.append(['export', '{}={}'.format(k, env_vars[k])])
    for kv in self.env_vars:
      export_cmd.append(['export', '{}'.format(kv)])
    return export_cmd

  def _prepare_scripts(self, cmd):
    worker_script_map = {}
    for i in range(len(self._cluster.get_client_workers())):
      script_path = self.SCRIPT_PATH_TMPL.format(pid=os.getpid(), worker=i)

      # ex. script = [['conda', 'activate', 'pytorch'], ['python3', 'train.py']]
      script = []
      script.extend(self._env_vars_cmd(i))
      # Setup environment for non-interactive non-login shell over ssh
      script.append(['.', '/etc/profile'])
      if self.tpuvm_mode:
        # Start the local tf server if it is not already running.
        script.append([
            'python3', '-m', self.XRT_RUN_SERVER_CMD, '--port',
            str(self.tpuvm_server_port)
        ])
        if self.restart_server:
          script[-1].append('--restart')
      if self.docker_image:
        script.append(self._docker_run_cmd(cmd))
      else:
        if self.conda_env:
          script.append(['conda', 'activate', self.conda_env])
        script.append(cmd)

      # ex. script_body = 'conda activate pytorch; python3 train.py'
      script_cmd_list = [concat_cmd_list(command) for command in script]
      script_body = concat_cmd_list(script_cmd_list, delimiter='; ')
      os.makedirs(os.path.dirname(script_path), exist_ok=True)
      with open(script_path, 'w') as f:
        f.write(script_body)
      subprocess.call(['chmod', '+x', script_path])
      worker_script_map[self._cluster.get_client_workers()[i]] = {
          'local_path':
              script_path,
          'remote_path':
              os.path.join('{}-remote'.format(os.path.dirname(script_path)),
                           os.path.basename(script_path)),
      }

    return worker_script_map

  def _scp_scripts(self, script_map):

    def _gcloud_scp(local_path, remote_path, client_worker):
      self._build_and_run_ssh(
          ['mkdir', '-p', os.path.dirname(remote_path)], client_worker)
      scp_cmd = self._build_scp_cmd(local_path, remote_path, client_worker)
      proc = subprocess.Popen(
          scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      self._stream_logs(proc, client_worker)

    threads = []
    for i, client_worker in enumerate(script_map):
      local_path = script_map[client_worker]['local_path']
      remote_path = script_map[client_worker]['remote_path']
      if i == 0:
        # ssh keygen single time
        _gcloud_scp(local_path, remote_path, client_worker)
        continue
      thread = threading.Thread(
          target=_gcloud_scp,
          daemon=True,
          args=(
              local_path,
              remote_path,
              client_worker,
          ))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def _cleanup(self, script_map):

    def _cleanup_worker(local_script, remote_script, client_worker):
      rm_tmp_dir = ['rm', '-rf', os.path.dirname(remote_script)]
      self._build_and_run_ssh(rm_tmp_dir, client_worker, log=False)
      subprocess.call(['rm', '-rf', os.path.dirname(local_script)])
      if self.docker_image:
        rm_container = ['docker', 'rm', '-f', self.docker_container]
        self._build_and_run_ssh(rm_container, client_worker, log=False)
      rm_pgroup = (
          'kill -9 -$(ps xao pid,pgid,cmd | grep "bash -c \\"{}\\""'
          r' | grep -v grep | awk "{{print \$2}}")').format(remote_script)
      self._build_and_run_ssh(rm_pgroup, client_worker, log=False)

    threads = []
    for client_worker in script_map:
      thread = threading.Thread(
          target=_cleanup_worker,
          args=(
              script_map[client_worker]['local_path'],
              script_map[client_worker]['remote_path'],
              client_worker,
          ))
      thread.start()
      threads.append(thread)

    # Cleanup states in case of restart
    self._initialize()

    for thread in threads:
      thread.join()

  def _start_run(self, script_map):

    def _run_script(script_paths, client_worker):
      script_path = script_paths['remote_path']
      exit_code = self._build_and_run_ssh([script_path], client_worker)
      if exit_code != 0:
        raise RuntimeError(
            'Remote command exitted with code: {}'.format(exit_code))

    def _regular_health_check():
      uneven_health_timeout = xu.getenv_as('XLA_UNEVEN_HEARTBEAT_TIMEOUT', int,
                                           900)
      even_health_timeout = xu.getenv_as('XLA_EVEN_HEARTBEAT_TIMEOUT', int,
                                         1800)
      while True:
        self._check_client_mesh_health(uneven_health_timeout,
                                       even_health_timeout)
        time.sleep(self.HEARTBEAT_CHECK_PERIOD)

    threading.Thread(target=_regular_health_check, daemon=True).start()
    xu.parallel_work(
        len(script_map), _run_script, script_map.values(), script_map.keys())

  def _run_cmd(self, script_map):
    try:
      self._scp_scripts(script_map)
      self._start_run(script_map)
    except KeyboardInterrupt:
      self.logger.warning(
          'Child process received Ctrl^C. Exiting...',
          extra={
              'clientip': '',
              'ordinal': ''
          })
      sys.exit(128 + signal.SIGINT)

  def run(self, cmd):
    self.trials = 0
    while self.trials <= self.MAX_TPU_RETRY:
      try:
        self.logger.info(
            'Command to distribute: {}'.format(concat_cmd_list(cmd)),
            extra={
                'clientip': '',
                'ordinal': ''
            })
        self.logger.info(
            f'Cluster configuration: {self._cluster}',
            extra={
                'clientip': '',
                'ordinal': ''
            })

        script_map = self._prepare_scripts(cmd)
        proc = multiprocessing.Process(target=self._run_cmd, args=(script_map,))
        proc.start()
        while True:
          if not proc.is_alive():
            sys.exit(proc.exitcode)
          if len(self._cluster.list_tpus_with_health(
              'UNHEALTHY_MAINTENANCE')) != 0:
            # TPU Maintenance: kill all training, wait for healthy, and restart
            break
          if not self._error_queue.empty():
            # Potential HostError on GCE VM: kill all, wait, and restart
            self.logger.warning(
                self._error_queue.get(), extra={
                    'clientip': '',
                    'ordinal': ''
                })
            break

          proc.join(10)

        # First wait for VMs to come back then cleanup all others
        self._cluster.wait_for_healthy_client(self)
        self._cleanup(script_map)
        proc.terminate()
        self._cluster.wait_for_healthy_service()
        self.trials += 1
      except KeyboardInterrupt:
        self.logger.info(
            'Cleaning up processes (takes a couple of seconds)',
            extra={
                'clientip': '',
                'ordinal': ''
            })
        self._cleanup(script_map)
        sys.exit(128 + signal.SIGINT)

    self.logger.info(
        'Max number of retries reached.', extra={
            'clientip': '',
            'ordinal': ''
        })


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='PyTorch on TPU distrubuted training launcher.',
      epilog=('Usage example: python3 -m'
              ' torch_xla.distributed.xla_dist --tpu=[TPU_NAME]'
              ' --conda-env torch-xla-nightly -- python3 train.py'))

  cluster_group = parser.add_argument_group('Cluster Setup')
  cluster_group.add_argument(
      '--tpu', type=str, required=True, help='Name of the Cloud TPU pod.')
  cluster_group.add_argument(
      '--vm',
      action='append',
      type=str,
      help=('List of single Compute VM instance names. '
            'If not provided we assume usage of instance groups.'))

  docker_group = parser.add_argument_group('Docker Setup')
  docker_group.add_argument(
      '--docker-container',
      default='',
      type=str,
      help='Name of docker container if running in docker.')
  docker_group.add_argument(
      '--docker-image',
      default='',
      type=str,
      help='Name of docker image if running in container.')
  docker_group.add_argument(
      '--docker-run-flag',
      action='append',
      type=str,
      help='Docker run flags to run container with (ex. --shm-size, ...).')

  conda_group = parser.add_argument_group('Conda Setup')
  conda_group.add_argument(
      '--conda-env',
      default='',
      type=str,
      help='Name of the conda environment if running with conda.')

  parser.add_argument(
      '--env',
      action='append',
      type=str,
      help='List of environment variables to distribute.')
  parser.add_argument(
      '--restart-tpuvm-pod-server',
      action='store_true',
      help='Restart the long running XRT local service for this training.')
  parser.add_argument(
      '--tpuvm-server-port',
      default=51011,
      type=int,
      help='Port that XRT local service will be start on.')
  parser.add_argument(
      'positional',
      nargs='+',
      type=str,
      help='The python command to launch training including model parameters.')

  FLAGS = parser.parse_args()

  if (FLAGS.docker_container or FLAGS.docker_image or
      FLAGS.docker_run_flag) and FLAGS.conda_env:
    raise ValueError('Docker Setup arguments and Conda Setup'
                     ' arguments are mutually exclusive.')

  # Resolve VM and TPU clusters.
  cluster_resolver = ClusterResolver(FLAGS.tpu, vms=FLAGS.vm)
  cluster = cluster_resolver.get_cluster()
  tpuvm_mode = cluster_resolver.get_tpuvm_mode()
  executor = DistributedExecutor(
      cluster,
      docker_container=FLAGS.docker_container,
      docker_image=FLAGS.docker_image,
      docker_run_flags=FLAGS.docker_run_flag,
      conda_env=FLAGS.conda_env,
      env_vars=FLAGS.env,
      restart_server=FLAGS.restart_tpuvm_pod_server,
      tpuvm_mode=tpuvm_mode,
      tpuvm_server_port=FLAGS.tpuvm_server_port)
  executor.run(FLAGS.positional)
