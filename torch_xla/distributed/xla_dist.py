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
from datetime import datetime, timedelta
from collections import namedtuple
import threading

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
  MESH_SERVICE_PORT = 8477  # Use single port to disallow concurrent runs
  DIST_ENV_VARS = [
      'XRT_TPU_CONFIG',
      'XRT_LOCAL_WORKER',
      'XRT_MESH_SERVICE_ADDRESS',
      'XRT_SHARD_WORLD_SIZE',
      'XRT_SHARD_ORDINAL',
      'XLA_EMIT_STEPLOG',
  ]
  DEFAULT_CONTAINER_NAME = 'pytorchtpudistrunner'
  MAX_TPU_RETRY = 50
  HEARTBEAT_CHECK_PERIOD = 30
  DIR_SYNC_FILESIZE_TOL = 0.02
  DIR_SYNC_TIMEDELTA_MIN = 10

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
      }
      for cw in self._cluster.get_client_workers()
    }
    self._error_queue = multiprocessing.Queue()
    self._last_heartbeat_check_time = 0

  def __init__(self,
               cluster,
               docker_container=None,
               docker_image=None,
               docker_run_flags=None,
               conda_env=None,
               checkpoint_sync_dir=None,
               env_vars=None):
    self._cluster = cluster
    self._initialize()
    client_master_ip = ClusterResolver.get_instance_metadata(
      'instance/network-interfaces/0/ip')
    self._client_master = next(filter(
      lambda cw: cw.get_internal_ip() == client_master_ip,
      self._cluster.get_client_workers()))
    self.logger = self._get_logger()
    self.docker_container = docker_container or self.DEFAULT_CONTAINER_NAME
    self.docker_image = docker_image
    self.docker_run_flags = list(docker_run_flags) if docker_run_flags else None
    self.conda_env = conda_env
    self.env_vars = list(env_vars) if env_vars else []
    self.checkpoint_sync_dir = checkpoint_sync_dir

    for env_var in self.env_vars:
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

  def _check_client_mesh_health(
      self, uneven_health_timeout=300, even_health_timeout=1800):
    max_delay = 0.0
    count = None
    now = time.time()
    for cw_hb in self._last_heartbeats.values():
      max_delay = max(max_delay, now - cw_hb['last_time'])
      if count is None:
        count = cw_hb['count']
      elif count >= 0 and count != cw_hb['count']:
        count = -1

    if count < 0 and max_delay > uneven_health_timeout:
      self._error_queue.put(RuntimeError(
        'Client mesh is unhealthy with uneven heartbeats'))
    elif count > 0 and max_delay > even_health_timeout:
      self._error_queue.put(RuntimeError(
        'Client mesh is unhealthy with even heartbeats'))

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
        log_fn(
            std_line,
            extra={
                'clientip': client_ip,
                'ordinal': ordinal
            })

    stdout = threading.Thread(
        target=_stream_output, daemon=True, args=(
            process.stdout,
            self.logger.info,
        ))
    stdout.start()
    stderr = threading.Thread(
        target=_stream_output, daemon=True, args=(
            process.stderr,
            self.logger.error,
        ))
    stderr.start()
    stdout.join()
    stderr.join()

  def _build_scp_cmd(
      self, local_path, remote_path, client_worker, recurse=False):
    cmd = ['gcloud', '-q', 'compute', 'scp']
    if recurse:
      cmd.append('--recurse')
    cmd.extend(['--internal-ip', '--zone={}'.format(client_worker.get_zone())])
    if not isinstance(local_path, list):
        local_path = [local_path]
    cmd.extend(local_path)
    cmd.append('{}:{}'.format(client_worker.get_hostname(), remote_path))
    return cmd

  def _build_ssh_cmd(self, remote_cmd, client_worker):
    if isinstance(remote_cmd, list):
      remote_cmd = concat_cmd_list(remote_cmd)
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

  def _run_remote_cmd(self, cmd, client_worker, shell=True, log=True):
    cmd = concat_cmd_list(cmd, quote='') if shell else cmd
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
    if log:
      self._stream_logs(proc, client_worker)
    proc.wait()
    if proc.returncode == 255:
      self._error_queue.put(RuntimeError(
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
      key = re.match('(\w*)=.*', env_kv)
      if key:
        docker_cmd.extend(['-e', key.group(1)])
    docker_cmd.append(self.docker_image)
    docker_cmd.extend(cmd)
    return docker_cmd

  def _env_vars_cmd(self, worker_idx):
    client_worker = self._cluster.get_client_workers()[worker_idx]
    env_vars = {
        'XRT_LOCAL_WORKER':
            'c_tpu_worker:{}'.format(worker_idx),
        'XRT_MESH_SERVICE_ADDRESS':
            '{}:{}'.format(
              self._client_master.get_internal_ip(), self.MESH_SERVICE_PORT),
        'XRT_SHARD_WORLD_SIZE':
            len(self._cluster.get_client_workers()),
        'XRT_SHARD_ORDINAL':
            worker_idx,
        'XLA_EMIT_STEPLOG': 1,
    }
    # Only for master
    if client_worker == self._client_master:
      xrt_server_config = [
          'c_tpu_worker;{worker_idx};{worker_ip}:{worker_port}'.format(
              worker_idx=idx,
              worker_ip=service_worker.get_internal_ip(),
              worker_port=service_worker.get_port())
          for idx, service_worker in enumerate(
            self._cluster.get_service_workers())
      ]
      xrt_tpu_config = '|'.join(xrt_server_config)
      env_vars['XRT_TPU_CONFIG'] = '{}'.format(xrt_tpu_config)

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

  def _gcloud_scp(self, local_path, remote_path, client_worker):
    self._build_and_run_ssh(
        ['mkdir', '-p', os.path.dirname(remote_path)], client_worker)
    scp_cmd = self._build_scp_cmd(local_path, remote_path, client_worker)
    proc = subprocess.Popen(
        scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    self._stream_logs(proc, client_worker)

  def _scp_scripts(self, script_map):
    threads = []
    for i, client_worker in enumerate(script_map):
      local_path = script_map[client_worker]['local_path']
      remote_path = script_map[client_worker]['remote_path']
      if i == 0:
        # ssh keygen single time
        _gcloud_scp(local_path, remote_path, client_worker)
        continue
      thread = threading.Thread(
          target=_gcloud_scp, daemon=True, args=(
              local_path,
              remote_path,
              client_worker,
          ))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def _sync_checkpoints(self):
    """
    Used to synchronize checkpoints between workers, before restarting
      the training script.
    For the time being, assumes master's copy is healthy.
    Assumes user has implemented logic to load checkpoints safely in their code.

    * get the last modified file/dir in path
    * get the files/dirs that are within 10 mins of that file/dir.
    * get sizes for the same files from remote hosts
    * compare sizes
    * for all remote hosts that has a drastically different copy
      * scp over from healthy host
    """

    def find_healthy_host(dir_states):
      # At the moment, assumes master is always healthy
      for state, host in dir_states:
        if host == self._client_master:
          return state, host

    def parse_stat_output(stdout):
      lines = stdout.decode("utf-8").split('\n')
      dat = [line.split() for line in lines if line]
      file_info_dict = {}
      for line_dat in dat:
        filename = ' '.join(line_dat[:-3])
        size = float(line_dat.pop())
        timestamp = datetime.strptime(
            ' '.join(line_dat[-2:]), "%Y-%m-%d %H:%M:%S")
        file_info_dict[filename] = fileinfo(timestamp, size)
      return file_info_dict

    def parse_du_output(stdout):
      lines = stdout.decode("utf-8").split('\n')
      dat = [line.split() for line in lines if line]
      update_dir_sizes_dict = {}
      for line_dat in dat:
        size = float(line_dat[0])
        filename = ' '.join(line_dat[1:])
        if filename == self.checkpoint_sync_dir:
          continue
        update_dir_sizes_dict[filename] = size
      return update_dir_sizes_dict

    def mkdir_and_get_info(client_worker):
      command = ';'.join([
          'mkdir -p {path} ',
          'find {path} -mindepth 1 -maxdepth 1 | xargs stat -c "%n %.19y %s"',
      ]).format(path=self.checkpoint_sync_dir)
      command = self._build_ssh_cmd(command, client_worker)
      command = concat_cmd_list(command, quote='')
      stdout, stderr = subprocess.Popen(
          command, shell=True, stdout=subprocess.PIPE,
          stderr=subprocess.PIPE).communicate()
      file_info_dict = parse_stat_output(stdout)
      # stat command does not print the full directory size.
      # use `du` to correct directory sizes.
      du_command = 'du --max-depth 1 -b {path}'.format(
          path=self.checkpoint_sync_dir)
      du_command = self._build_ssh_cmd(du_command, client_worker)
      du_command = concat_cmd_list(du_command, quote='')
      stdout, stderr = subprocess.Popen(
          du_command, shell=True, stdout=subprocess.PIPE).communicate()
      update_dir_sizes_dict = parse_du_output(stdout)
      update_dir_sizes_dict = {
          filename: fileinfo(file_info_dict[filename].timestamp, size)
          for filename, size in update_dir_sizes_dict.items()}
      file_info_dict.update(update_dir_sizes_dict)
      return file_info_dict, client_worker

    def mkdirs_and_get_infos():
      workers = self._cluster._client_workers
      return xu.parallel_work(len(workers), mkdir_and_get_info, workers)

    def get_last_checkpoint_files(healthy_dir_info):
      last_modified_ts, _ = max(healthy_dir_info.values())
      after_ts = last_modified_ts - timedelta(self.DIR_SYNC_TIMEDELTA_MIN)
      last_chpt_files = {
          filename: fileinfo
          for filename, fileinfo in healthy_dir_info.items()
          if fileinfo.timestamp > after_ts}
      return last_chpt_files

    def is_host_corrupt(last_checkpoint_files, dir_info):
      corrupt = False
      for filename, fileinfo in last_checkpoint_files.items():
        if filename not in dir_info:
          return True
        elif fileinfo.size == 0:
          continue
        remote_size = dir_info[filename].size
        size_delta = fileinfo.size - remote_size
        if size_delta / fileinfo.size > self.DIR_SYNC_FILESIZE_TOL:
          return True
      return False

    def get_corrupt_hosts(last_checkpoint_files, dir_infos):
      return [host for (dir_info, host) in dir_infos
              if is_host_corrupt(last_checkpoint_files, dir_info)]

    def copy_from_healthy_host(
        healthy_host, last_checkpoint_files, remote_host):
      # Assuming master is healthy, copying from local files.
      local_paths = list(last_checkpoint_files.keys())
      remote_path = self.checkpoint_sync_dir
      scp_cmd = self._build_scp_cmd(
          local_paths, remote_path, remote_host, recurse=True)
      proc = subprocess.Popen(
          scp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      proc.wait()

    if self.checkpoint_sync_dir is None:
      return
    self.logger.info(
        'Begin syncing checkpoints at "{}"'.format(self.checkpoint_sync_dir),
        extra={'clientip': '', 'ordinal': ''})
    fileinfo = namedtuple('fileinfo', ['timestamp', 'size'])
    dir_infos = mkdirs_and_get_infos()
    healthy_dir_info, healthy_host = find_healthy_host(dir_infos)
    if not healthy_dir_info:
      return
    last_checkpoint_files = get_last_checkpoint_files(healthy_dir_info)
    corrupt_hosts = get_corrupt_hosts(last_checkpoint_files, dir_infos)
    if corrupt_hosts:
        self.logger.info(
            'Found corrupt checkpoints in following hosts: {}'.format(
                ', '.join(host.get_internal_ip() for host in corrupt_hosts)),
            extra={'clientip': '', 'ordinal': ''})
        args = [(healthy_host, last_checkpoint_files, host)
                for host in corrupt_hosts]
        xu.parallel_work(
            len(corrupt_hosts), copy_from_healthy_host, *zip(*args))
        self.logger.info(
            'Synchronized checkpoints', extra={'clientip': '', 'ordinal': ''})

  def _cleanup(self, script_map):

    def _cleanup_worker(local_script, remote_script, client_worker):
      rm_tmp_dir = ['rm', '-rf', os.path.dirname(remote_script)]
      self._build_and_run_ssh(rm_tmp_dir, client_worker, log=False)
      subprocess.call(['rm', '-rf', os.path.dirname(local_script)])
      if self.docker_image:
        rm_container = ['docker', 'rm', '-f', self.docker_container]
        self._build_and_run_ssh(rm_container, client_worker, log=False)
      rm_pgroup = ('kill -9 -$(ps xao pid,pgid,cmd | grep "bash -c \\"{}\\""'
                   ' | grep -v grep | awk "{{print \$2}}")').format(
                     remote_script)
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

    def _run_script(script_path, client_worker):
      self._build_and_run_ssh([script_path], client_worker)

    def _regular_health_check():
      while True:
        self._check_client_mesh_health()
        time.sleep(self.HEARTBEAT_CHECK_PERIOD)

    threads = []
    for client_worker in script_map:
      thread = threading.Thread(
          target=_run_script,
          daemon=True,
          args=(
              script_map[client_worker]['remote_path'],
              client_worker,
          ))
      thread.start()
      threads.append(thread)

    threading.Thread(target=_regular_health_check, daemon=True).start()
    for thread in threads:
      thread.join()

  def _run_cmd(self, script_map):
    try:
      self._scp_scripts(script_map)
      self._start_run(script_map)
    except KeyboardInterrupt:
      self.logger.warning(
        'Child process received Ctrl^C. Exiting...',
        extra={'clientip': '', 'ordinal': ''})
      sys.exit(128 + signal.SIGINT)

  def run(self, cmd):
    trials = 0
    while trials <= self.MAX_TPU_RETRY:
      try:
        self.logger.info(
            'Command to distribute: {}'.format(concat_cmd_list(cmd)),
            extra={'clientip': '', 'ordinal': ''})
        self.logger.info(
            f'Cluster configuration: {self._cluster}',
            extra={'clientip': '', 'ordinal': ''})

        script_map = self._prepare_scripts(cmd)
        self._sync_checkpoints()
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
              self._error_queue.get(), extra={'clientip': '', 'ordinal': ''})
            break

          proc.join(10)

        # First wait for VMs to come back then cleanup all others
        self._cluster.wait_for_healthy_client(self)
        self._cleanup(script_map)
        proc.terminate()
        self._cluster.wait_for_healthy_service()
        trials += 1
      except KeyboardInterrupt:
        self.logger.info(
          'Cleaning up processes (takes a couple of seconds)',
          extra={'clientip': '', 'ordinal': ''})
        self._cleanup(script_map)
        sys.exit(128 + signal.SIGINT)

    self.logger.info(
      'Max number of retries reached.', extra={'clientip': '', 'ordinal': ''})

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='PyTorch on TPU distrubuted training launcher.',
      epilog=('Usage example: python -m'
              ' torch_xla.distributed.xla_dist --tpu=[TPU_NAME]'
              ' --conda-env torch-xla-nightly -- python train.py'))

  cluster_group = parser.add_argument_group('Cluster Setup')
  cluster_group.add_argument(
      '--tpu',
      type=str,
      required=True,
      help='Name of the Cloud TPU pod.')
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
      '--checkpoint-sync-dir',
      type=str,
      default=None,
      help='''Path to the directory to sync between workers at initialization,
              as well as automatic restarts.''')
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
  executor = DistributedExecutor(
      cluster,
      docker_container=FLAGS.docker_container,
      docker_image=FLAGS.docker_image,
      docker_run_flags=FLAGS.docker_run_flag,
      conda_env=FLAGS.conda_env,
      checkpoint_sync_dir=FLAGS.checkpoint_sync_dir,
      env_vars=FLAGS.env)
  executor.run(FLAGS.positional)
