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


def update_tpu_runtime(tpu_ip, version):
  print(f'Updating TPU runtime to {version.tpu} ...')
  url = 'http://{tpu_ip}:8475/requestversion/{tpu_version}'.format(
      tpu_ip=tpu_ip, tpu_version=version.tpu)
  print('Done updating TPU runtime: {}'.format(requests.post(url)))


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


def parse_env_tpu_ip():
  # In both Colab and Kaggle: TPU_NAME='grpc://abc.def.ghi.jkl:8470'
  tpu_addr = os.environ.get('TPU_NAME', None)
  tpu_ip_regex = re.compile('grpc://(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):8470')
  m_tpu_ip = tpu_ip_regex.match(tpu_addr)
  if not m_tpu_ip:
    raise ValueError('TPU not found.')

  return m_tpu_ip.group(1)


def install_vm(version, apt_packages):
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
  tpu_ip = args.tpu_ip if args.tpu_ip else parse_env_tpu_ip()

  # Update TPU
  print('Updating TPU and VM. This may take around 2 minutes.')
  update = threading.Thread(
      target=update_tpu_runtime, args=(
          tpu_ip,
          version,
      ))
  update.start()
  install_vm(version, args.apt_packages)
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
      '--tpu-ip',
      type=str,
      help='TPU internal ip address',
  )
  args = parser.parse_args()
  run_setup(args)
