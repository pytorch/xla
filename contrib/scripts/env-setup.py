#!/usr/bin/env python
# Sample usage:
#   python env-setup.py --version xrt==1.15.0 --apt-packages libomp5
import argparse
import collections
from datetime import datetime, timedelta
import os
import re
import requests
import subprocess
import threading


VersionConfig = collections.namedtuple('VersionConfig', ['wheels', 'tpu'])
STABLE_VERSION_MAP = {
  'xrt==1.15.0': VersionConfig('1.15', '1.15.0'),
}
DIST_BUCKET = 'gs://tpu-pytorch/wheels'
TORCH_WHEEL_TMPL = 'torch-{whl_version}-cp36-cp36m-linux_x86_64.whl'
TORCH_XLA_WHEEL_TMPL = 'torch_xla-{whl_version}-cp36-cp36m-linux_x86_64.whl'
TORCHVISION_WHEEL_TMPL = 'torchvision-{whl_version}-cp36-cp36m-linux_x86_64.whl'


def update_tpu_runtime(tpu_ip, version):
  print('Updating TPU runtime to {} ...'.format(args.version))
  url = 'http://{tpu_ip}:8475/requestversion/{tpu_version}'.format(
    tpu_ip=tpu_ip, tpu_version=version.tpu)
  print('Done updating TPU runtime: {}'.format(requests.post(url)))

def get_version(version):
  if version == 'nightly':
    return VersionConfig('nightly', 'TPU-dev{}'.format(
      (datetime.today() - timedelta(1)).strftime('%Y%m%d')))

  try:
    datetime.strptime(version, '%Y%m%d')
    return VersionConfig(f'nightly+{version}', f'TPU-dev{version}')
  except ValueError:
    pass  # Not a dated nightly.

  if version not in STABLE_VERSION_MAP:
    raise ValueError(f'Version {version} unknown')
  return STABLE_VERSION_MAP[version]

def parse_env_tpu_ip():
  # In both Colab and Kaggle: TPU_NAME='grpc://abc.def.ghi.jkl:8470'
  tpu_addr = os.environ.get('TPU_NAME', None)
  tpu_ip_regex = re.compile('grpc://(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):8470')
  m_tpu_ip = tpu_ip_regex.match(tpu_addr)
  if not m_tpu_ip:
    raise ValueError('TPU not found.')

  return m_tpu_ip.group(1)

def install_vm(version, apt_packages):
  torch_whl = TORCH_WHEEL_TMPL.format(whl_version=version.wheels)
  torch_whl_path = os.path.join(DIST_BUCKET, torch_whl)
  torch_xla_whl = TORCH_XLA_WHEEL_TMPL.format(whl_version=version.wheels)
  torch_xla_whl_path = os.path.join(DIST_BUCKET, torch_xla_whl)
  torchvision_whl = TORCHVISION_WHEEL_TMPL.format(whl_version=version.wheels)
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
  update = threading.Thread(target=update_tpu_runtime, args=(tpu_ip, version,))
  update.start()
  install_vm(version, args.apt_packages)
  update.join()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--version',
      type=str,
      default='nightly',
      help='Versions to install (nightly, release version, or YYYYMMDD).',
  )
  parser.add_argument(
      '--apt-packages',
      nargs='+',
      default=['libomp5', 'sox', 'libsox-dev'],
      help='List of apt packages to install',
  )
  parser.add_argument(
    '--tpu-ip',
    type=str,
    help='TPU internal ip address',
  )
  args = parser.parse_args()
  run_setup(args)
