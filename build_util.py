import os
from typing import Iterable
import subprocess
import sys
import shutil


def check_env_flag(name: str, default: str = '') -> bool:
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def bazel_options_from_env() -> Iterable[str]:
  bazel_flags = []

  if check_env_flag('DEBUG'):
    bazel_flags.append('--config=dbg')

  if check_env_flag('TPUVM_MODE'):
    bazel_flags.append('--config=tpu')

  gcloud_key_file = os.getenv('GCLOUD_SERVICE_KEY_FILE', default='')
  # Remote cache authentication.
  if gcloud_key_file:
    # Temporary workaround to allow PRs from forked repo to run CI. See details at (#5259).
    # TODO: Remove the check once self-hosted GHA workers are available to CPU/GPU CI.
    gcloud_key_file_size = os.path.getsize(gcloud_key_file)
    if gcloud_key_file_size > 1:
      bazel_flags.append('--google_credentials=%s' % gcloud_key_file)
      bazel_flags.append('--config=remote_cache')
  else:
    if check_env_flag('BAZEL_REMOTE_CACHE'):
      bazel_flags.append('--config=remote_cache')

  cache_silo_name = os.getenv('SILO_NAME', default='dev')
  if cache_silo_name:
    bazel_flags.append('--remote_default_exec_properties=cache-silo-key=%s' %
                       cache_silo_name)

  bazel_jobs = os.getenv('BAZEL_JOBS', default='')
  if bazel_jobs:
    bazel_flags.append('--jobs=%s' % bazel_jobs)

  # Build configuration.
  if check_env_flag('BAZEL_VERBOSE'):
    bazel_flags.append('-s')
  if check_env_flag('XLA_CUDA'):
    bazel_flags.append('--config=cuda')
  if check_env_flag('XLA_CPU_USE_ACL'):
    bazel_flags.append('--config=acl')

  return bazel_flags


def bazel_build(bazel_target: str,
                destination_dir: str,
                options: Iterable[str] = []):
  bazel_argv = [
      'bazel', 'build', bazel_target,
      f"--symlink_prefix={os.path.join(os.getcwd(), 'bazel-')}"
  ]

  # Remove duplicated flags because they confuse bazel
  flags = set(bazel_options_from_env() + options)
  bazel_argv.extend(flags)

  print(' '.join(bazel_argv), flush=True)
  subprocess.check_call(bazel_argv, stdout=sys.stdout, stderr=sys.stderr)

  target_path = bazel_target.replace('@xla//', 'external/xla/').replace(
      '//', '').replace(':', '/')
  output_path = os.path.join('bazel-bin', target_path)
  output_filename = os.path.basename(output_path)

  if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

  shutil.copyfile(output_path, os.path.join(destination_dir, output_filename))
