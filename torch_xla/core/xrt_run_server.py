import argparse
import re
import time
import os
import subprocess
import sys

from pathlib import Path
from torch_xla.__init__ import server_is_alive, XRT_RUN_SERVER_PROCESS, XRT_SERVER_REGEX


def kill_service():
  subprocess.call(['pkill', '-f', XRT_SERVER_REGEX])
  # Wait unitl existing server process gets killed.
  found_server_process = False
  while server_is_alive():
    found_server_process = True
    time.sleep(1)
  # Server process might still hold the lock to the tpu device after turing into a zombie
  # process with name <defunct>. Sleep a bit longer to make sure it exit completely.
  if found_server_process:
    time.sleep(5)


def run_service(port, flag_env):
  if server_is_alive():
    print('Server is already running, use --restart(--restart-tpuvm-pod-server '
          'if running with xla_dist) to restart the server.')
    return

  local_env = os.environ.copy()
  # Enable the basic logging by defualt
  local_env['TF_CPP_MIN_LOG_LEVEL'] = '0'
  local_env[
      'TF_CPP_VMODULE'] = 'tpu_configuration_ops=1,tpu_execute_op=1,tpu_compile_op=1,tpu_compile_op_impl=1,tpu_compile_op_common=1,tpu_compile_ops=1,master=1,computation_client=5'

  env_vars = list(flag_env) if flag_env else []
  for env_var in env_vars:
    if re.match(r'\w*=\w*', env_var) is None:
      raise ValueError(('Environment variable to distribute ({}) should follow '
                        'the form: X=Y').format(env_var))
    (env, var) = env_var.split('=', 1)
    local_env[env] = var

  Path('/tmp/xrt_server_log').mkdir(parents=True, exist_ok=True)
  time_str = time.strftime('%Y%m%d-%H%M%S')
  log_file = open('/tmp/xrt_server_log/server_{}.log'.format(time_str), 'w')
  subprocess.Popen(['python3', '-m', XRT_RUN_SERVER_PROCESS,
                    str(port)],
                   stdout=log_file,
                   stderr=subprocess.STDOUT,
                   env=local_env,
                   start_new_session=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--port', type=int, help='Port that XRT local service will be using.')
  parser.add_argument(
      '--env',
      action='append',
      type=str,
      help='List of environment variables to distribute.')

  server_state_group = parser.add_mutually_exclusive_group()
  server_state_group.add_argument(
      '--restart',
      action='store_true',
      help='Restart the long running XRT local server.')
  server_state_group.add_argument(
      '--stop',
      action='store_true',
      help='Stop the long running XRT local server.')

  FLAGS = parser.parse_args()
  if FLAGS.restart or FLAGS.stop:
    kill_service()

  if not FLAGS.stop:
    run_service(FLAGS.port, FLAGS.env)
