#!/usr/bin/env python
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import re
import subprocess
import sys
import threading


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
  MASTER_IDX = 0
  MESH_SERVICE_PORT = 8477  # Use single port to disallow concurrent runs
  DIST_ENV_VARS = [
      'XRT_TPU_CONFIG',
      'XRT_LOCAL_WORKER',
      'XRT_MESH_SERVICE_ADDRESS',
      'XRT_SHARD_WORLD_SIZE',
      'XRT_SHARD_ORDINAL',
  ]
  DEFAULT_CONTAINER_NAME = 'pytorchtpudistrunner'

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
    self.docker_run_flags = list(docker_run_flags) if docker_run_flags else None
    self.conda_env = conda_env
    self.env_vars = list(env_vars) if env_vars else []

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
        '{}:{}'.format(client_worker._hostname, remote_path),
    ]

  def _build_ssh_cmd(self, remote_cmd, client_worker):
    if isinstance(remote_cmd, list):
      remote_cmd = concat_cmd_list(remote_cmd)
    return [
        'gcloud',
        '-q',
        'compute',
        'ssh',
        '--internal-ip',
        '--zone={}'.format(client_worker._zone),
        '{}'.format(client_worker._hostname),
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
            '{}:{}'.format(client_master._internal_ip, self.MESH_SERVICE_PORT),
        'XRT_SHARD_WORLD_SIZE':
            len(self._cluster._client_workers),
        'XRT_SHARD_ORDINAL':
            worker_idx,
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
      env_vars['XRT_TPU_CONFIG'] = '{}'.format(xrt_tpu_config)

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
      script_cmd_list = [concat_cmd_list(command) for command in script]
      script_body = concat_cmd_list(script_cmd_list, delimiter='; ')
      os.makedirs(os.path.dirname(script_path), exist_ok=True)
      with open(script_path, 'w') as f:
        f.write(script_body)
      subprocess.call(['chmod', '+x', script_path])
      worker_script_map[self._cluster._client_workers[i]] = {
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
          target=_gcloud_scp, args=(
              local_path,
              remote_path,
              client_worker,
          ))
      thread.daemon = True
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def _cleanup(self, script_map):

    def _cleanup_worker(local_script, remote_script, client_worker):
      rm_tmp_dir = ['rm', '-rf', os.path.dirname(remote_script)]
      self._build_and_run_ssh(rm_tmp_dir, client_worker)
      subprocess.call(['rm', '-rf', os.path.dirname(local_script)])
      if self.docker_image:
        rm_container = ['docker', 'rm', '-f', self.docker_container]
        self._build_and_run_ssh(rm_container, client_worker)
      rm_pgroup = ('kill -9 -$(ps xao pid,pgid,cmd | grep {} | grep -v grep'
                   ' | awk "{{print \$2}}")').format(remote_script)
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
    for thread in threads:
      thread.join()

  def _start_run(self, script_map):

    def _run_script(script_path, client_worker):
      self._build_and_run_ssh([script_path], client_worker)

    threads = []
    for client_worker in script_map:
      thread = threading.Thread(
          target=_run_script,
          args=(
              script_map[client_worker]['remote_path'],
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
        'Command to distribute: {}'.format(concat_cmd_list(cmd)),
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
      env_vars=FLAGS.env)
  executor.run(FLAGS.positional)
