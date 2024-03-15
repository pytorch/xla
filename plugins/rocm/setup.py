import os
import sys

# add `build_util` to import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import build_util
import setuptools

build_util.bazel_build('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so',
                       'torch_xla_rocm_plugin/lib', ['--config=rocm'])

setuptools.setup()
