#!/usr/bin/env python
# Sample usage:
#   python env-setup.py --version 1.5 --apt-packages libomp5
import argparse
import collections
from datetime import datetime
import os
import platform
import re
import requests
import subprocess
import threading

VersionConfig = collections.namedtuple('VersionConfig',
                                       ['wheels', 'tpu', 'py_version'])
OLDEST_VERSION = datetime.strptime('20200318', '%Y%m%d')
DIST_BUCKET = 'gs://tpu-pytorch/wheels'
TORCH_WHEEL_TMPL = 'torch-{whl_version}-cp{py_version}-cp{py_version}m-linux_x86_64.whl'
TORCH_XLA_WHEEL_TMPL = 'torch_xla-{whl_version}-cp{py_version}-cp{py_version}m-linux_x86_64.whl'
TORCHVISION_WHEEL_TMPL = 'torchvision-{whl_version}-cp{py_version}-cp{py_version}m-linux_x86_64.whl'


def update_tpu_runtime(tpu_name, version):
  print(f'Updating TPU runtime to {version.tpu} ...')

  try:
    import cloud_tpu_client
  except ImportError:
    subprocess.call(['pip', 'install', 'cloud-tpu-client'])
    import cloud_tpu_client

  client = cloud_tpu_client.Client(tpu_name)
  client.configure_tpu_version(version.tpu)
  print('Done updating TPU runtime')


def get_py_version():
  version_tuple = platform.python_version_tuple()
  return version_tuple[0] + version_tuple[1]  # major_version + minor_version


def get_version(version):
  if version == 'nightly':
    return VersionConfig('nightly', 'pytorch-nightly', get_py_version())

  version_date = None
  try:
    version_date = datetime.strptime(version, '%Y%m%d')
  except ValueError:
    pass  # Not a dated nightly.

  if version_date:
    if version_date < OLDEST_VERSION:
      raise ValueError(f'Oldest nightly version available is {OLDEST_VERSION}')
    return VersionConfig(f'nightly+{version}', f'pytorch-dev{version}',
                         get_py_version())

  version_regex = re.compile('^(\d+\.)+\d+$')
  if not version_regex.match(version):
    raise ValueError(f'{version} is an invalid torch_xla version pattern')
  return VersionConfig(version, f'pytorch-{version}', get_py_version())


def install_vm(version, apt_packages, is_root=False):
  torch_whl = TORCH_WHEEL_TMPL.format(
      whl_version=version.wheels, py_version=version.py_version)
  torch_whl_path = os.path.join(DIST_BUCKET, torch_whl)
  torch_xla_whl = TORCH_XLA_WHEEL_TMPL.format(
      whl_version=version.wheels, py_version=version.py_version)
  torch_xla_whl_path = os.path.join(DIST_BUCKET, torch_xla_whl)
  torchvision_whl = TORCHVISION_WHEEL_TMPL.format(
      whl_version=version.wheels, py_version=version.py_version)
  torchvision_whl_path = os.path.join(DIST_BUCKET, torchvision_whl)
  apt_cmd = ['apt-get', 'install', '-y']
  apt_cmd.extend(apt_packages)

  if not is_root:
    # Colab/Kaggle run as root, but not GCE VMs so we need privilege
    apt_cmd.insert(0, 'sudo')

  installation_cmds = [
      ['pip', 'uninstall', '-y', 'torch', 'torchvision'],
      ['gsutil', 'cp', torch_whl_path, '.'],
      ['gsutil', 'cp', torch_xla_whl_path, '.'],
      ['gsutil', 'cp', torchvision_whl_path, '.'],
      ['pip', 'install', torch_whl],
      ['pip', 'install', torch_xla_whl],
      ['pip', 'install', torchvision_whl],
      apt_cmd,
  ]
  for cmd in installation_cmds:
    subprocess.call(cmd)


def run_setup(args):
  version = get_version(args.version)
  # Update TPU
  print('Updating TPU and VM. This may take around 2 minutes.')
  update = threading.Thread(
      target=update_tpu_runtime, args=(
          args.tpu,
          version,
      ))
  update.start()
  install_vm(version, args.apt_packages, is_root=not args.tpu)
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
