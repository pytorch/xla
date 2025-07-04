#!/usr/bin/env python
# Welcome to the PyTorch/XLA setup.py.
#
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with debug symbols
#
#   TORCH_XLA_VERSION
#     specify the version of PyTorch/XLA, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   GIT_VERSIONED_XLA_BUILD
#     creates a git versioned build
#
#   TORCH_XLA_PACKAGE_NAME
#     change the package name to something other than 'torch_xla'
#
#   BAZEL_VERBOSE=0
#     turn on verbose messages during the bazel build of the xla/xrt client
#
#   XLA_CUDA=0
#     build the xla/xrt client with CUDA enabled
#
#   XLA_CPU_USE_ACL=0
#     whether to use ACL
#
#   BUNDLE_LIBTPU=0
#     include libtpu in final wheel

#   BUILD_CPP_TESTS=0
#     build the C++ tests
#
#   GCLOUD_SERVICE_KEY_FILE=''
#     file containing the auth tokens for remote cache/build. implies remote cache.
#
#   BAZEL_REMOTE_CACHE=""
#     whether to use remote cache for builds
#
#   TPUVM_MODE=0
#     whether to build for TPU
#
#   SILO_NAME=""
#     name of the remote build cache silo
#
#   CXX_ABI=""
#     value for cxx_abi flag; if empty, it is inferred from `torch._C`.
#
from setuptools import setup, find_packages, distutils, Extension, command
from setuptools.command import develop, build_ext
import posixpath
import contextlib
import distutils.ccompiler
import distutils.command.clean
import os
import re
import requests
import shutil
import subprocess
import sys
import tempfile
import zipfile

import build_util

import platform

platform_machine = platform.machine()

base_dir = os.path.dirname(os.path.abspath(__file__))

# How to update libtpu and JAX to a new nightly build
# ===================================================
#
# Most of the time, you can just run
#   scripts/update_deps.py
# to automatically update the versions of openxla, libtpu, and jax used in
# torch_xla. In case the script doesn't work and you need to do it manually,
# follow instructions below.
#
# Usually we update both at the same time to minimize their version skew.
#
# 1. Update libtpu to a new nightly build:
#
#    1. Find a new version of libtpu on https://storage.googleapis.com/libtpu-wheels/index.html.
#       Usually we prefer the latest version.
#       E.g. libtpu/libtpu-0.0.16.dev20250529+nightly-py3-none-manylinux_2_31_x86_64.whl
#    2. Update _libtpu_version to the libtpu version. E.g. 0.0.16.
#    3. Update _libtpu_date to the date of the version. E.g. 20250529.
#
# 2. Update JAX to a new nightly build:
#
#    1. Find a new version of jax and jaxlib on https://storage.googleapis.com/jax-releases/jax_nightly_releases.html.
#       Usually we prefer the latest version.
#       E.g. jax/jax-0.6.1.dev20250428-py3-none-any.whl and nocuda/jaxlib-0.6.1.dev20250428-*.whl
#       Both jax and jaxlib should be on the same day. We prefer this date to be
#       the same as the libtpu build date, but it's not strictly necessary.
#    2. Update _jax_version and _jaxlib_version to the versions we found. E.g.
#       0.6.1.
#    3. Update _jax_date to the date of the new jax and jaxlib build. E.g. 20250428.
#
# 3. After updating libtpu and JAX, run
#      scripts/build_developer.sh
#    for a sanity check. Fix the build errors as needed.
#
# 4. After the local build succeeds, create a PR and wait for the CI result. Fix
#    CI errors as needed until all required checks pass.

USE_NIGHTLY = True  # Whether to use nightly or stable libtpu and JAX.

_libtpu_version = '0.0.18'
_libtpu_date = '20250617'

_jax_version = '0.6.2'
_jaxlib_version = '0.6.2'
_jax_date = '20250617'  # Date for jax and jaxlib.

if USE_NIGHTLY:
  _libtpu_version += f".dev{_libtpu_date}"
  _jax_version += f'.dev{_jax_date}'
  _jaxlib_version += f'.dev{_jax_date}'
  _libtpu_wheel_name = f'libtpu-{_libtpu_version}.dev{_libtpu_date}+nightly-py3-none-manylinux_2_31_{platform_machine}'
  _libtpu_storage_directory = 'libtpu-nightly-releases'
else:
  # The postfix can be changed when the version is updated. Check
  # https://storage.googleapis.com/libtpu-wheels/index.html for correct
  # versioning.
  _libtpu_wheel_name = f'libtpu-{_libtpu_version}-py3-none-manylinux_2_31_{platform_machine}'
  _libtpu_storage_directory = 'libtpu-lts-releases'

_libtpu_storage_path = f'https://storage.googleapis.com/{_libtpu_storage_directory}/wheels/libtpu/{_libtpu_wheel_name}.whl'


def _get_build_mode():
  for i in range(1, len(sys.argv)):
    if not sys.argv[i].startswith('-'):
      return sys.argv[i]


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


def get_build_version(xla_git_sha):
  version = os.getenv('TORCH_XLA_VERSION', '2.9.0')
  if build_util.check_env_flag('GIT_VERSIONED_XLA_BUILD', default='TRUE'):
    try:
      version += '+git' + xla_git_sha[:7]
    except Exception:
      pass
  return version


def create_version_files(base_dir, version, xla_git_sha, torch_git_sha):
  print('Building torch_xla version: {}'.format(version))
  print('XLA Commit ID: {}'.format(xla_git_sha))
  print('PyTorch Commit ID: {}'.format(torch_git_sha))
  py_version_path = os.path.join(base_dir, 'torch_xla', 'version.py')
  with open(py_version_path, 'w') as f:
    f.write('# Autogenerated file, do not edit!\n')
    f.write("__version__ = '{}'\n".format(version))
    f.write("__xla_gitrev__ = '{}'\n".format(xla_git_sha))
    f.write("__torch_gitrev__ = '{}'\n".format(torch_git_sha))

  cpp_version_path = os.path.join(base_dir, 'torch_xla', 'csrc', 'version.cpp')
  with open(cpp_version_path, 'w') as f:
    f.write('// Autogenerated file, do not edit!\n')
    f.write('#include "torch_xla/csrc/version.h"\n\n')
    f.write('namespace torch_xla {\n\n')
    f.write('const char XLA_GITREV[] = {{"{}"}};\n'.format(xla_git_sha))
    f.write('const char TORCH_GITREV[] = {{"{}"}};\n\n'.format(torch_git_sha))
    f.write('}  // namespace torch_xla\n')


def maybe_bundle_libtpu(base_dir):
  libtpu_path = os.path.join(base_dir, 'torch_xla', 'lib', 'libtpu.so')
  with contextlib.suppress(FileNotFoundError):
    os.remove(libtpu_path)

  if not build_util.check_env_flag('BUNDLE_LIBTPU', '0'):
    return

  try:
    import libtpu
    module_path = os.path.dirname(libtpu.__file__)
    print('Found pre-installed libtpu at ', module_path)
    shutil.copyfile(os.path.join(module_path, 'libtpu.so'), libtpu_path)
  except ModuleNotFoundError:
    print('No installed libtpu found. Downloading...')

    with tempfile.NamedTemporaryFile('wb') as whl:
      resp = requests.get(_libtpu_storage_path)
      resp.raise_for_status()

      whl.write(resp.content)
      whl.flush()

      os.makedirs(os.path.join(base_dir, 'torch_xla', 'lib'), exist_ok=True)
      with open(libtpu_path, 'wb') as libtpu_so:
        z = zipfile.ZipFile(whl.name)
        libtpu_so.write(z.read('libtpu/libtpu.so'))


class Clean(distutils.command.clean.clean):

  def bazel_clean_(self):
    self.spawn(['bazel', 'clean', '--expunge'])

  def run(self):
    import glob
    import re
    with open('.gitignore', 'r') as f:
      ignores = f.read()
      pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
      for wildcard in filter(None, ignores.split('\n')):
        match = pat.match(wildcard)
        if match:
          if match.group(1):
            # Marker is found and stop reading .gitignore.
            break
          # Ignore lines which begin with '#'.
        else:
          for filename in glob.glob(wildcard):
            try:
              os.remove(filename)
            except OSError:
              shutil.rmtree(filename, ignore_errors=True)

    self.execute(self.bazel_clean_, (), msg="Cleaning bazel outputs")

    # It's an old-style class in Python 2.7...
    distutils.command.clean.clean.run(self)


xla_git_sha, torch_git_sha = get_git_head_sha(base_dir)
version = get_build_version(xla_git_sha)

build_mode = _get_build_mode()
if build_mode not in ['clean']:
  # Generate version info (torch_xla.__version__).
  create_version_files(base_dir, version, xla_git_sha, torch_git_sha)

  # Copy libtpu.so into torch_xla/lib
  maybe_bundle_libtpu(base_dir)


class BazelExtension(Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        posixpath.relpath(bazel_target, '//').split(':'))
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
    command.build_ext.build_ext.run(self)  # type: ignore

  def bazel_build(self, ext):
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    bazel_argv = [
        'bazel', 'build', ext.bazel_target,
        f"--symlink_prefix={os.path.join(self.build_temp, 'bazel-')}"
    ]

    build_cpp_tests = build_util.check_env_flag('BUILD_CPP_TESTS', default='0')
    if build_cpp_tests:
      bazel_argv.append('//:cpp_tests')

    import torch
    cxx_abi = os.getenv('CXX_ABI') or getattr(torch._C,
                                              '_GLIBCXX_USE_CXX11_ABI', None)
    if cxx_abi is not None:
      bazel_argv.append(f'--cxxopt=-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}')

    bazel_argv.extend(build_util.bazel_options_from_env())

    self.spawn(bazel_argv)

    ext_bazel_bin_path = os.path.join(self.build_temp, 'bazel-bin', ext.relpath,
                                      ext.target_name)
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


# Read in README.md for our long_description
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

# Finds torch_xla and its subpackages
packages_to_include = find_packages(include=['torch_xla*'])
# Explicitly add torchax
packages_to_include.extend(find_packages(where='torchax', include=['torchax*']))

# Map the top-level 'torchax' package name to its source location
torchax_dir = os.path.join(cwd, 'torchax')
package_dir_mapping = {'torch_xla': os.path.join(cwd, 'torch_xla')}
package_dir_mapping['torchax'] = os.path.join(torchax_dir, 'torchax')


class Develop(develop.develop):

  def run(self):
    # Build the C++ extension
    self.run_command("build_ext")

    # Run the standard develop process first
    # This installs dependencies, scripts, and importantly, creates an `.egg-link` file
    super().run()

    # Replace the `.egg-link` with a `.pth` file.
    self.link_packages()

  def link_packages(self):
    """
    There are two mechanisms to install an "editable" package in Python: `.egg-link`
    and `.pth` files. setuptools uses `.egg-link` by default. However, `.egg-link`
    only supports linking a single directory containg one editable package.
    This function removes the `.egg-link` file and generates a `.pth` file that can
    be used to link multiple packages, in particular, `torch_xla` and `torchax`.

    Note that this function is only relevant in the editable package development path
    (`python setup.py develop`). Nightly and release wheel builds work out of the box
    without egg-link/pth.
    """
    # Ensure paths like self.install_dir are set
    self.ensure_finalized()

    # Get the site-packages directory
    target_dir = self.install_dir

    # Remove the standard .egg-link file
    # It's usually named based on the distribution name
    dist_name = self.distribution.get_name()
    egg_link_file = os.path.join(target_dir, dist_name + '.egg-link')
    if os.path.exists(egg_link_file):
      print(f"Removing default egg-link file: {egg_link_file}")
      try:
        os.remove(egg_link_file)
      except OSError as e:
        print(f"Warning: Could not remove {egg_link_file}: {e}")

    # Create our custom .pth file with specific paths
    cwd = os.path.dirname(__file__)
    # Path containing 'torch_xla' package source: ROOT
    path_for_torch_xla = os.path.abspath(cwd)
    # Path containing 'torchax' package source: ROOT/torchax
    path_for_torchax = os.path.abspath(os.path.join(cwd, 'torchax'))

    paths_to_add = {path_for_torch_xla, path_for_torchax}

    # Construct a suitable .pth filename (PEP 660 style is good practice)
    version = self.distribution.get_version()
    # Sanitize name and version for filename (replace runs of non-alphanumeric chars with '-')
    sanitized_name = re.sub(r"[^a-zA-Z0-9.]+", "_", dist_name)
    sanitized_version = re.sub(r"[^a-zA-Z0-9.]+", "_", version)
    pth_filename = os.path.join(
        target_dir, f"__editable_{sanitized_name}_{sanitized_version}.pth")

    # Ensure site-packages exists
    os.makedirs(target_dir, exist_ok=True)

    # Write the paths to the .pth file, one per line
    with open(pth_filename, "w", encoding='utf-8') as f:
      for path in sorted(paths_to_add):
        f.write(path + "\n")


def _get_jax_install_requirements():
  if not USE_NIGHTLY:
    # Stable versions of JAX can be directly installed from PyPI.
    return [
        f'jaxlib=={_jaxlib_version}',
        f'jax=={_jax_version}',
    ]

  # Install nightly JAX libraries from the JAX package registries.
  jax = f'jax @ https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/jax/jax-{_jax_version}-py3-none-any.whl'

  jaxlib = []
  for python_minor_version in [9, 10, 11, 12]:
    jaxlib.append(
        f'jaxlib @ https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/jaxlib/jaxlib-{_jaxlib_version}-cp3{python_minor_version}-cp3{python_minor_version}-manylinux2014_x86_64.whl ; python_version == "3.{python_minor_version}"'
    )
  return [jax] + jaxlib


setup(
    name=os.environ.get('TORCH_XLA_PACKAGE_NAME', 'torch_xla'),
    version=version,
    description='XLA bridge for PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pytorch/xla',
    author='PyTorch/XLA Dev Team',
    author_email='pytorch-xla@googlegroups.com',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10.0",
    packages=packages_to_include,
    package_dir=package_dir_mapping,
    ext_modules=[
        BazelExtension('//:_XLAC.so'),
        BazelExtension('//:_XLAC_cuda_functions.so'),
    ],
    install_requires=[
        'absl-py>=1.0.0',
        'numpy',
        'pyyaml',
        'requests',
        # importlib.metadata backport required for PJRT plugin discovery prior
        # to Python 3.10
        'importlib_metadata>=4.6;python_version<"3.10"',
        # Some torch operations are lowered to HLO via JAX.
        *_get_jax_install_requirements(),
    ],
    package_data={
        'torch_xla': ['lib/*.so*',],
    },
    entry_points={
        'console_scripts': [
            'stablehlo-to-saved-model = torch_xla.tf_saved_model_integration:main'
        ],
        'torch_xla.plugins': [
            'tpu = torch_xla._internal.tpu:TpuPlugin',
            'neuron = torch_xla._internal.neuron:NeuronPlugin',
            'xpu = torch_xla._internal.xpu:XpuPlugin'
        ],
    },
    extras_require={
        # On Cloud TPU VM install with:
        # pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-wheels/index.html -f https://storage.googleapis.com/libtpu-releases/index.html
        'tpu': [
            f'libtpu=={_libtpu_version}',
            'tpu-info',
        ],
        # As of https://github.com/pytorch/xla/pull/8895, jax is always a dependency of torch_xla.
        # However, this no-op extras_require entrypoint is left here for backwards compatibility.
        # pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
        'pallas': [f'jaxlib=={_jaxlib_version}', f'jax=={_jax_version}'],
    },
    cmdclass={
        'build_ext': BuildBazelExtension,
        'clean': Clean,
        'develop': Develop,
    })
