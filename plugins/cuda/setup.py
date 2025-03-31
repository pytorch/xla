import datetime
import os
import sys

# add `build_util` to import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import build_util
import setuptools

# Fix C++ compiler flags for the build
bazel_flags = [
    '--config=cuda',
    '--action_env=CC=/usr/bin/gcc-12',
    '--action_env=CXX=/usr/bin/g++-12',
    '--host_copt=-I/usr/include', 
    '--host_copt=-I/usr/lib/gcc/x86_64-linux-gnu/12/include',
    '--copt=-I/usr/include',
    '--copt=-I/usr/lib/gcc/x86_64-linux-gnu/12/include',
    # Add include directories explicitly
    '--copt=-fPIC',
    '--cxxopt=-std=c++17'
]

build_util.bazel_build('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so',
                       'torch_xla_cuda_plugin/lib', bazel_flags)
setuptools.setup(
    # TODO: Use a common version file
    version=os.getenv('TORCH_XLA_VERSION',
                      f'2.8.0.dev{datetime.date.today().strftime("%Y%m%d")}'))
