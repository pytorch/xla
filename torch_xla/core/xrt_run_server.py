import argparse
import re
import time
import os
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--port', type=str, help='Port that XRT local service will be using.')
  parser.add_argument(
      '--env',
      action='append',
      type=str,
      help='List of environment variables to distribute.')
  FLAGS = parser.parse_args()

  my_env = os.environ.copy()
  # Enable the basic logging by defualt
  my_env['TF_CPP_MIN_LOG_LEVEL'] = '0'
  my_env[
      'TF_CPP_VMODULE'] = 'tpu_configuration_ops=1,tpu_execute_op=1,tpu_compile_op=1,tpu_compile_op_impl=1,tpu_compile_op_common=1,tpu_compile_ops=1,master=1,computation_client=5'

  env_vars = list(FLAGS.env) if FLAGS.env else []
  for env_var in env_vars:
    if re.match(r'\w*=\w*', env_var) is None:
      raise ValueError(('Environment variable to distribute ({}) should follow '
                        'the form: X=Y').format(env_var))
    (env, var) = env_var.split('=', 1)
    my_env[env] = var

  Path("/tmp/xrt_server_log").mkdir(parents=True, exist_ok=True)
  time_str = time.strftime("%Y%m%d-%H%M%S")
  stderr_file = open("/tmp/xrt_server_log/server_err_{}.log".format(time_str),
                     "w")
  stdout_file = open("/tmp/xrt_server_log/server_out_{}.log".format(time_str),
                     "w")
  subprocess.Popen(
      ["python", "-m", "torch_xla.core._xrt_run_server", FLAGS.port],
      stdout=stdout_file,
      stderr=stderr_file,
      env=my_env,
      start_new_session=True)
