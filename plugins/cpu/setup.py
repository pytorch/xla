import os
import sys

# add `build_util` to import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import build_util
import setuptools

build_util.bazel_build('//plugins/cpu:pjrt_c_api_cpu_plugin.so',
                       'torch_xla_cpu_plugin/lib')

setuptools.setup()
