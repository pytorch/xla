import subprocess
import sys
from typing import Iterable
import setuptools
import shutil
import os

def _check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

DEBUG = _check_env_flag('DEBUG')
GCLOUD_KEY_FILE = os.getenv('GCLOUD_SERVICE_KEY_FILE', default='')
CACHE_SILO_NAME = os.getenv('SILO_NAME', default='dev')
BAZEL_JOBS = os.getenv('BAZEL_JOBS', default='')

def _bazel_build(bazel_target: str, destination_path: str, options: Iterable[str] = []):
  bazel_argv = ['bazel', 'build', bazel_target, f"--symlink_prefix={os.path.join(os.getcwd(), 'bazel-')}"]
  # for opt in extra_compile_args:
  #   bazel_argv.append("--cxxopt={}".format(opt))

  # Debug build.
  if DEBUG:
    bazel_argv.append('--config=dbg')

  # Remote cache authentication.
  if GCLOUD_KEY_FILE:
    # Temporary workaround to allow PRs from forked repo to run CI. See details at (#5259).
    # TODO: Remove the check once self-hosted GHA workers are available to CPU/GPU CI.
    gcloud_key_file_size = os.path.getsize(GCLOUD_KEY_FILE)
    if gcloud_key_file_size > 1:
      bazel_argv.append('--google_credentials=%s' % GCLOUD_KEY_FILE)
      bazel_argv.append('--config=remote_cache')
  else:
    if _check_env_flag('BAZEL_REMOTE_CACHE'):
      bazel_argv.append('--config=remote_cache')
  if CACHE_SILO_NAME:
    bazel_argv.append('--remote_default_exec_properties=cache-silo-key=%s' %
                      CACHE_SILO_NAME)

  if BAZEL_JOBS:
    bazel_argv.append('--jobs=%s' % BAZEL_JOBS)

  # Build configuration.
  if _check_env_flag('BAZEL_VERBOSE'):
    bazel_argv.append('-s')

  bazel_argv.extend(options)

  # subprocess.check_call(['pwd'], stdout=sys.stdout, stderr=sys.stderr)
  subprocess.check_call(bazel_argv, stdout=sys.stdout, stderr=sys.stderr)

  target_path = bazel_target.replace('@xla//', 'external/xla/').replace(':', '/')
  output_path = os.path.join('bazel-bin', target_path)
  output_filename = os.path.basename(output_path)

  if not os.path.exists(destination_path):
      os.makedirs(destination_path)

  shutil.copyfile(output_path, os.path.join(destination_path, output_filename))

_bazel_build('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so', 'torch_xla_cuda_plugin/lib', ['--config=cuda'])

setuptools.setup()
