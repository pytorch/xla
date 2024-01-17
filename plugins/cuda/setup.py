import setuptools
from setuptools import Extension
from setuptools.command import build_ext
import shutil
import os
import posixpath
import torch


def _check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


extra_compile_args = []
cxx_abi = os.getenv(
    'CXX_ABI', default='') or getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', None)
if cxx_abi is not None:
  extra_compile_args.append(f'-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}')


DEBUG = _check_env_flag('DEBUG')
GCLOUD_KEY_FILE = os.getenv('GCLOUD_SERVICE_KEY_FILE', default='')
CACHE_SILO_NAME = os.getenv('SILO_NAME', default='dev')
BAZEL_JOBS = os.getenv('BAZEL_JOBS', default='')

class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        bazel_target.replace('@xla//', 'external/xla/').split(':'))
    print('searchforthisstring', self.relpath)
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
    if ext_name.endswith('.so'):
      ext_name = ext_name[:-3]
    Extension.__init__(self, ext_name, sources=[])


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    for ext in self.extensions:
      self.bazel_build(ext)
    build_ext.build_ext.run(self)

  def bazel_build(self, ext):
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    bazel_argv = [
        'bazel', 'build', ext.bazel_target,
        f"--symlink_prefix={os.path.join('plugins/cuda', self.build_temp, 'bazel-')}"
    ]
    for opt in extra_compile_args:
      bazel_argv.append("--cxxopt={}".format(opt))

    # Debug build.
    if DEBUG:
      bazel_argv.append('--config=dbg')

    # Remote cache authentication.
    if GCLOUD_KEY_FILE:
      # Temporary workaround to allow PRs from forked repo to run CI. See details at (#5259).
      # TODO: Remove the check once self-hosted GHA workers are available to CPU/GPU CI.
      gclout_key_file_size = os.path.getsize(GCLOUD_KEY_FILE)
      if gclout_key_file_size > 1:
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

    bazel_argv.append('--config=cuda')

    self.spawn(bazel_argv)

    ext_bazel_bin_path = os.path.join(self.build_temp, 'bazel-bin', ext.relpath,
                                      ext.target_name)
    print(ext_bazel_bin_path)
    ext_dest_path = self.get_ext_fullpath(ext.name)
    print(ext_dest_path)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)

setuptools.setup(
    url='https://github.com/pytorch/xla',
    ext_modules=[
        BazelExtension('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so'),
    ],
    cmdclass={
        'build_ext': BuildBazelExtension,
})
