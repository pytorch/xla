import os
from collections.abc import Iterable
import subprocess
import sys
import shutil
from dataclasses import dataclass
import functools

import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@functools.lru_cache
def get_pinned_packages():
  """Gets the versions of important pinned dependencies of torch_xla."""
  return PinnedPackages(
      use_nightly=True,
      date='20250424',
      raw_libtpu_version='0.0.14',
      raw_jax_version='0.6.1',
      raw_jaxlib_version='0.6.1',
  )


@functools.lru_cache
def get_build_version():
  xla_git_sha, _torch_git_sha = get_git_head_sha(BASE_DIR)
  version = os.getenv('TORCH_XLA_VERSION', '2.8.0')
  if check_env_flag('GIT_VERSIONED_XLA_BUILD', default='TRUE'):
    try:
      version += '+git' + xla_git_sha[:7]
    except Exception:
      pass
  return version


@functools.lru_cache
def get_git_head_sha(base_dir):
  xla_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        cwd=base_dir).decode('ascii').strip()
  if os.path.isdir(os.path.join(base_dir, '..', '.git')):
    torch_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=os.path.join(
                                                base_dir,
                                                '..')).decode('ascii').strip()
  else:
    torch_git_sha = ''
  return xla_git_sha, torch_git_sha


@functools.lru_cache
def get_jax_install_requirements():
  """Get a list of JAX requirements for use in setup.py without extra package registries."""
  pinned_packages = get_pinned_packages()
  if not pinned_packages.use_nightly:
    # Stable versions of JAX can be directly installed from PyPI.
    return [
        f'jaxlib=={pinned_packages.jaxlib_version}',
        f'jax=={pinned_packages.jax_version}',
    ]

  # Install nightly JAX libraries from the JAX package registries.
  # TODO(https://github.com/pytorch/xla/issues/9064): This URL needs to be
  # updated to use the new JAX package registry for any JAX builds after Apr 28, 2025.
  jax = f'jax @ https://storage.googleapis.com/jax-releases/nightly/jax/jax-{pinned_packages.jax_version}-py3-none-any.whl'
  jaxlib = []
  for python_minor_version in [9, 10, 11]:
    jaxlib.append(
        f'jaxlib @ https://storage.googleapis.com/jax-releases/nightly/nocuda/jaxlib-{pinned_packages.jaxlib_version}-cp3{python_minor_version}-cp3{python_minor_version}-manylinux2014_x86_64.whl ; python_version == "3.{python_minor_version}"'
    )
  return [jax] + jaxlib


@functools.lru_cache
def get_jax_cuda_requirements():
  """Get a list of JAX CUDA requirements for use in setup.py without extra package registries."""
  pinned_packages = get_pinned_packages()
  jax_requirements = get_jax_install_requirements()

  # Install nightly JAX CUDA libraries.
  jax_cuda = [
      f'jax-cuda12-pjrt @ https://storage.googleapis.com/jax-releases/nightly/wheels/jax_cuda12_pjrt-{pinned_packages.jax_version}-py3-none-manylinux2014_x86_64.whl'
  ]
  for python_minor_version in [9, 10, 11]:
    jax_cuda.append(
        f'jax-cuda12-plugin @ https://storage.googleapis.com/jax-releases/nightly/wheels/jax_cuda12_plugin-{pinned_packages.jaxlib_version}-cp3{python_minor_version}-cp3{python_minor_version}-manylinux2014_x86_64.whl ; python_version == "3.{python_minor_version}"'
    )

  return jax_requirements + jax_cuda


@dataclass(eq=True, frozen=True)
class PinnedPackages:
  use_nightly: bool
  """Whether to use nightly or stable libtpu and JAX"""

  date: str
  """The date of the libtpu and jax build"""

  raw_libtpu_version: str
  """libtpu version string in [major].[minor].[patch] format."""

  raw_jax_version: str
  """jax version string in [major].[minor].[patch] format."""

  raw_jaxlib_version: str
  """jaxlib version string in [major].[minor].[patch] format."""

  @property
  def libtpu_version(self) -> str:
    if self.use_nightly:
      return f'{self.raw_libtpu_version}.dev{self.date}'
    else:
      return self.raw_libtpu_version

  @property
  def jax_version(self) -> str:
    if self.use_nightly:
      return f'{self.raw_jax_version}.dev{self.date}'
    else:
      return self.raw_jax_version

  @property
  def jaxlib_version(self) -> str:
    if self.use_nightly:
      return f'{self.raw_jaxlib_version}.dev{self.date}'
    else:
      return self.raw_jaxlib_version

  @property
  def libtpu_storage_directory(self) -> str:
    if self.use_nightly:
      return 'libtpu-nightly-releases'
    else:
      return 'libtpu-lts-releases'

  @property
  def libtpu_wheel_name(self) -> str:
    if self.use_nightly:
      return f'libtpu-{self.libtpu_version}+nightly'
    else:
      return f'libtpu-{self.libtpu_version}'

  @property
  def libtpu_storage_path(self) -> str:
    platform_machine = platform.machine()
    # The suffix can be changed when the version is updated. Check
    # https://storage.googleapis.com/libtpu-wheels/index.html for correct name.
    suffix = f"py3-none-manylinux_2_31_{platform_machine}"
    return f'https://storage.googleapis.com/{self.libtpu_storage_directory}/wheels/libtpu/{self.libtpu_wheel_name}-{suffix}.whl'


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
  flags = set(list(bazel_options_from_env()) + list(options))
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
