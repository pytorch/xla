#!/usr/bin/env python

from __future__ import print_function

from setuptools import setup, find_packages, distutils
from torch.utils.cpp_extension import BuildExtension, CppExtension
import distutils.ccompiler
import distutils.command.clean
import glob
import inspect
import multiprocessing
import multiprocessing.pool
import os
import platform
import re
import shutil
import subprocess
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))


def _check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def _compile_parallel(self,
                      sources,
                      output_dir=None,
                      macros=None,
                      include_dirs=None,
                      debug=0,
                      extra_preargs=None,
                      extra_postargs=None,
                      depends=None):
  # those lines are copied from distutils.ccompiler.CCompiler directly
  macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
      output_dir, macros, include_dirs, sources, depends, extra_postargs)
  cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

  def compile_one(obj):
    try:
      src, ext = build[obj]
    except KeyError:
      return
    self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

  list(
      multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()).imap(
          compile_one, objects))
  return objects


# Plant the parallel compile function.
if _check_env_flag('COMPILE_PARALLEL', default='1'):
  try:
    if (inspect.signature(distutils.ccompiler.CCompiler.compile) ==
        inspect.signature(_compile_parallel)):
      distutils.ccompiler.CCompiler.compile = _compile_parallel
  except:
    pass


class Clean(distutils.command.clean.clean):

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

    # It's an old-style class in Python 2.7...
    distutils.command.clean.clean.run(self)


class Build(BuildExtension):

  def run(self):
    # Run the original BuildExtension first. We need this before building
    # the tests.
    BuildExtension.run(self)
    # Build the C++ tests
    cmd = [os.path.join(base_dir, 'test/cpp/run_tests.sh'), '-B']
    if subprocess.call(cmd) != 0:
      print('Failed to build tests: {}'.format(cmd), file=sys.stderr)
      sys.exit(1)


# Generate the code before globbing!
generate_code_cmd = [os.path.join(base_dir, 'scripts', 'generate_code.sh')]
if subprocess.call(generate_code_cmd) != 0:
  print("Failed to run '{}'".format(generate_code_cmd), file=sys.stderr)
  sys.exit(1)

build_libs_cmd = [os.path.join(base_dir, 'build_torch_xla_libs.sh')]
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
  build_libs_cmd += [sys.argv[1]]
if subprocess.call(build_libs_cmd) != 0:
  print("Failed to run '{}'".format(build_libs_cmd), file=sys.stderr)
  sys.exit(1)

# Fetch the sources to be built.
torch_xla_sources = (
    glob.glob('torch_xla/csrc/*.cpp') + glob.glob('torch_xla/csrc/ops/*.cpp') +
    glob.glob('torch_xla/csrc/passes/*.cpp'))

# Constant known variables used throughout this file
lib_path = os.path.join(base_dir, 'torch_xla', 'lib')
pytorch_source_path = os.getenv('PYTORCH_SOURCE_PATH',
                                os.path.dirname(base_dir))
third_party_path = os.path.join(base_dir, 'third_party')

include_dirs = [
    base_dir,
]
include_dirs += [
    third_party_path + '/tensorflow/bazel-tensorflow',
    third_party_path + '/tensorflow/bazel-genfiles',
    third_party_path +
    '/tensorflow/bazel-tensorflow/external/protobuf_archive/src',
    third_party_path + '/tensorflow/bazel-tensorflow/external/eigen_archive',
    third_party_path + '/tensorflow/bazel-tensorflow/external/com_google_absl',
]
include_dirs += [
    pytorch_source_path,
    os.path.join(pytorch_source_path, 'torch', 'csrc'),
    os.path.join(pytorch_source_path, 'torch', 'lib', 'tmp_install', 'include'),
]

library_dirs = []
library_dirs.append(lib_path)

extra_link_args = []

DEBUG = _check_env_flag('DEBUG')
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')


def make_relative_rpath(path):
  if IS_DARWIN:
    return '-Wl,-rpath,@loader_path/' + path
  elif IS_WINDOWS:
    return ''
  else:
    return '-Wl,-rpath,$ORIGIN/' + path


extra_compile_args = [
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
]

if re.match(r'clang', os.getenv('CC', '')):
  extra_compile_args += [
      '-Wno-macro-redefined',
      '-Wno-return-std-move',
  ]

if DEBUG:
  if IS_WINDOWS:
    extra_link_args.append('/DEBUG:FULL')
  else:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']

extra_link_args += ['-lxla_computation_client']

version = os.getenv('TORCH_XLA_VERSION', '0.1')
if _check_env_flag('VERSIONED_XLA_BUILD', default='0'):
  try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                  cwd=base_dir).decode('ascii').strip()
    version += '+' + sha[:7]
  except Exception:
    pass

setup(
    name='torch_xla',
    version=version,
    description='XLA bridge for PyTorch',
    url='https://github.com/pytorch/xla',
    author='Alex Suhan, Davide Libenzi',
    author_email='asuhan@google.com, dlibenzi@google.com',
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    ext_modules=[
        CppExtension(
            '_XLAC',
            torch_xla_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + \
                [make_relative_rpath('torch_xla/lib')],
        ),
    ],
    package_data={
        'torch_xla': [
            'lib/*.so*',
        ],
    },
    data_files=[
        'test/cpp/build/test_ptxla',
        'scripts/fixup_binary.py',
    ],
    cmdclass={
        'build_ext': Build,
        'clean': Clean,
    })
