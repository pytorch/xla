import os
from typing import Iterable

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

  if check_env_flag('BUILD_CPP_TESTS', default='0'):
    bazel_flags.append('//test/cpp:all')
    bazel_flags.append('//torch_xla/csrc/runtime:all')

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
