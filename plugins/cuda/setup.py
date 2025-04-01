import os
import sys

# add `build_util` to import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import build_util
import setuptools

build_util.bazel_build('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so',
                       'torch_xla_cuda_plugin/lib', ['--config=cuda'])

setuptools.setup(
    version=build_util.get_build_version(),
    install_requires=build_util.get_jax_cuda_requirements(),
)
