#!/usr/bin/env python
# Sample usage:
#   python env-setup.py --version 1.11 --apt-packages libomp5
import argparse
import collections
from datetime import datetime
import os
import platform
import re
import requests
import subprocess
import threading
import sys

VersionConfig = collections.namedtuple('VersionConfig',
                                       ['wheels', 'tpu', 'py_version', 'cuda_version'])
DEFAULT_CUDA_VERSION = '11.2'
OLDEST_VERSION = datetime.strptime('20200318', '%Y%m%d')
NEW_VERSION = datetime.strptime('20220315', '%Y%m%d')  # 1.11 release date
OLDEST_GPU_VERSION = datetime.strptime('20200707', '%Y%m%d')
DIST_BUCKET = 'gs://tpu-pytorch/wheels'
TORCH_WHEEL_TMPL = 'torch-{whl_version}-cp{py_version}-cp{py_version}-linux_x86_64.whl'
TORCH_XLA_WHEEL_TMPL = 'torch_xla-{whl_version}-cp{py_version}-cp{py_version}-linux_x86_64.whl'
TORCHVISION_WHEEL_TMPL = 'torchvision-{whl_version}-cp{py_version}-cp{py_version}-linux_x86_64.whl'
VERSION_REGEX = re.compile(r'^(\d+\.)+\d+$')

def is_gpu_runtime():
  return os.environ.get('COLAB_GPU', 0) == '1'


def is_tpu_runtime():
  return 'TPU_NAME' in os.environ


def update_tpu_runtime(tpu_name, version):
  print(f'Updating TPU runtime to {version.tpu} ...')

  try:
    import cloud_tpu_client
  except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'cloud-tpu-client'])
    import cloud_tpu_client

  client = cloud_tpu_client.Client(tpu_name)
  client.configure_tpu_version(version.tpu)
  print('Done updating TPU runtime')


def get_py_version():
  version_tuple = platform.python_version_tuple()
  return version_tuple[0] + version_tuple[1]  # major_version + minor_version


def get_cuda_version():
  if is_gpu_runtime():
    # cuda available, install cuda wheels
    return DEFAULT_CUDA_VERSION


def get_version(version):
  cuda_version = get_cuda_version()
  if version == 'nightly':
    return VersionConfig(
      'nightly', 'pytorch-nightly', get_py_version(), cuda_version)

  version_date = None
  try:
    version_date = datetime.strptime(version, '%Y%m%d')
  except ValueError:
    pass  # Not a dated nightly.

  if version_date:
    if cuda_version and version_date < OLDEST_GPU_VERSION:
      raise ValueError(
        f'Oldest nightly version build with CUDA available is {OLDEST_GPU_VERSION}')
    elif not cuda_version and version_date < OLDEST_VERSION:
      raise ValueError(f'Oldest nightly version available is {OLDEST_VERSION}')
    return VersionConfig(f'nightly+{version}', f'pytorch-dev{version}',
                         get_py_version(), cuda_version)


  if not VERSION_REGEX.match(version):
    raise ValueError(f'{version} is an invalid torch_xla version pattern')
  return VersionConfig(
    version, f'pytorch-{version}', get_py_version(), cuda_version)


def install_vm(version, apt_packages, is_root=False):
  dist_bucket = DIST_BUCKET

  if version.cuda_version:
    # Distributions for GPU runtime
    # Note: GPU wheels available from 1.11
    dist_bucket = os.path.join(
      DIST_BUCKET, 'cuda/{}'.format(version.cuda_version.replace('.', '')))
  else:
    # Distributions for TPU runtime
    # Note: this redirection is required for 1.11 & nightly releases
    # because the current 2 VM wheels are not compatible with colab environment.
    if version.wheels == 'nightly':
      dist_bucket = os.path.join(DIST_BUCKET, 'colab/')
    elif 'nightly+' in version.wheels:
      build_date = datetime.strptime( version.wheels.split('+')[1], '%Y%m%d')
      if build_date >= NEW_VERSION:
        dist_bucket = os.path.join(DIST_BUCKET, 'colab/')
    elif VERSION_REGEX.match(version.wheels):
      minor = int(version.wheels.split('.')[1])
      if minor >= 11:
        dist_bucket = os.path.join(DIST_BUCKET, 'colab/')
    else:
      raise ValueError(f'{version} is an invalid torch_xla version pattern')

  torch_whl = TORCH_WHEEL_TMPL.format(
      whl_version=version.wheels, py_version=version.py_version)
  torch_whl_path = os.path.join(dist_bucket, torch_whl)
  torch_xla_whl = TORCH_XLA_WHEEL_TMPL.format(
      whl_version=version.wheels, py_version=version.py_version)
  torch_xla_whl_path = os.path.join(dist_bucket, torch_xla_whl)
  torchvision_whl = TORCHVISION_WHEEL_TMPL.format(
      whl_version=version.wheels, py_version=version.py_version)
  torchvision_whl_path = os.path.join(dist_bucket, torchvision_whl)
  apt_cmd = ['apt-get', 'install', '-y']
  apt_cmd.extend(apt_packages)

  if not is_root:
    # Colab/Kaggle run as root, but not GCE VMs so we need privilege
    apt_cmd.insert(0, 'sudo')

  installation_cmds = [
      [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision'],
      ['gsutil', 'cp', torch_whl_path, '.'],
      ['gsutil', 'cp', torch_xla_whl_path, '.'],
      ['gsutil', 'cp', torchvision_whl_path, '.'],
      [sys.executable, '-m', 'pip', 'install', torch_whl],
      [sys.executable, '-m', 'pip', 'install', torch_xla_whl],
      [sys.executable, '-m', 'pip', 'install', torchvision_whl],
      apt_cmd,
  ]
  for cmd in installation_cmds:
    subprocess.call(cmd)


def run_setup(args):
  version = get_version(args.version)
  # Update TPU
  print('Updating... This may take around 2 minutes.')

  if is_tpu_runtime():
    update = threading.Thread(
        target=update_tpu_runtime, args=(
            args.tpu,
            version,
        ))
    update.start()

  install_vm(version, args.apt_packages, is_root=not args.tpu)

  if is_tpu_runtime():
    update.join()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--version',
      type=str,
      default='20200515',
      help='Versions to install (nightly, release version, or YYYYMMDD).',
  )
  parser.add_argument(
      '--apt-packages',
      nargs='+',
      default=['libomp5'],
      help='List of apt packages to install',
  )
  parser.add_argument(
      '--tpu',
      type=str,
      help='[GCP] Name of the TPU (same zone, project as VM running script)',
  )
  args = parser.parse_args()
  run_setup(args)
