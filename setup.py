#!/usr/bin/env python

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import platform
import subprocess
import sys


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

torch_xla_sources = [
    'torch_xla/csrc/batch_norm.cpp',
    'torch_xla/csrc/convolution.cpp',
    'torch_xla/csrc/cross_replica_reduces.cpp',
    'torch_xla/csrc/data_ops.cpp',
    'torch_xla/csrc/elementwise.cpp',
    'torch_xla/csrc/graph_context.cpp',
    'torch_xla/csrc/helpers.cpp',
    'torch_xla/csrc/init_python_bindings.cpp',
    'torch_xla/csrc/log_softmax.cpp',
    'torch_xla/csrc/module.cpp',
    'torch_xla/csrc/nll_loss.cpp',
    'torch_xla/csrc/pooling.cpp',
    'torch_xla/csrc/reduction.cpp',
    'torch_xla/csrc/tensor.cpp',
    'torch_xla/csrc/torch_util.cpp',
    'torch_xla/csrc/translator.cpp',
    'torch_xla/csrc/passes/eval_static_size.cpp',
    'torch_xla/csrc/passes/insert_explicit_expand.cpp',
    'torch_xla/csrc/passes/remove_unused_forward_outputs.cpp',
    'torch_xla/csrc/passes/replace_untraced_operators.cpp',
    'torch_xla/csrc/passes/set_mat_mul_output_shape.cpp',
    'torch_xla/csrc/passes/threshold_backward_peephole.cpp',
]

build_libs_cmd = './build_torch_xla_libs.sh'

if subprocess.call(build_libs_cmd) != 0:
    print("Failed to run '{}'".format(build_libs_cmd))
    sys.exit(1)

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, 'torch_xla', 'lib')
pytorch_source_path = os.getenv('PYTORCH_SOURCE_PATH', '..')
third_party_path = os.path.join(cwd, 'third_party')

include_dirs = [
    third_party_path + '/tensorflow/bazel-tensorflow',
    third_party_path + '/tensorflow/bazel-genfiles',
    third_party_path + '/tensorflow/bazel-tensorflow/external/protobuf_archive/src',
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

extra_compile_args = []

if DEBUG:
    if IS_WINDOWS:
        extra_link_args.append('/DEBUG:FULL')
    else:
        extra_compile_args += ['-O0', '-g']
        extra_link_args += ['-O0', '-g']

extra_link_args += ['-lxla_computation_client']

version = '0.1'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
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
            '_C',
            torch_xla_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + [make_relative_rpath('torch_xla/lib')],
        ),
    ],
    package_data={
        'torch_xla': [
            'lib/*.so*',
        ]
    },
    cmdclass={'build_ext': BuildExtension})
