import subprocess
import sys
from typing import Iterable
import setuptools
import shutil
import os

import build_util

def _bazel_build(bazel_target: str, destination_path: str, options: Iterable[str] = []):
  bazel_argv = ['bazel', 'build', bazel_target, f"--symlink_prefix={os.path.join(os.getcwd(), 'bazel-')}"]

  # Remove duplicated flags because they confuse bazel
  flags = set(build_util.bazel_options_from_env() + options)
  bazel_argv.extend(flags)

  print(' '.join(bazel_argv), flush=True)
  subprocess.check_call(bazel_argv, stdout=sys.stdout, stderr=sys.stderr)

  target_path = bazel_target.replace('@xla//', 'external/xla/').replace(':', '/')
  output_path = os.path.join('bazel-bin', target_path)
  output_filename = os.path.basename(output_path)

  if not os.path.exists(destination_path):
      os.makedirs(destination_path)

  shutil.copyfile(output_path, os.path.join(destination_path, output_filename))

_bazel_build('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so', 'torch_xla_cuda_plugin/lib', ['--config=cuda'])

setuptools.setup()
