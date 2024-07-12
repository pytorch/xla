import functools
import glob
from ipaddress import ip_address
import operator
import os
import pathlib
import re
import socket
from typing import NamedTuple, Optional, List
from typing_extensions import TypedDict
import requests
import yaml

import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import plugins
from torch_xla.version import __version__

_GCE_METADATA_ROOT_URL = 'http://metadata.google.internal/computeMetadata/v1'
_ACCELERATOR_TYPE_TO_HOST_BOUNDS = {
    # v2
    'v2-8': '1,1,1',
    'v2-32': '2,2,1',
    'v2-128': '4,4,1',
    'v2-256': '4,8,1',
    'v2-512': '8,8,1',
    # v3
    'v3-8': '1,1,1',
    'v3-32': '2,2,1',
    'v3-64': '2,4,1',
    'v3-128': '4,4,1',
    'v3-256': '4,8,1',
    'v3-512': '8,8,1',
    'v3-1024': '8,16,1',
    'v3-2048': '16,16,1',
    # Get v4 host bounds from TPU metadata
}

_GOOGLE_PCI_VENDOR_ID = '0x1ae0'
_TPU_PCI_DEVICE_IDS = [
    # TPU v2, v3
    '0x0027',
    # TPU v4
    '0x005e',
    # TPU v5e
    '0x0063',
    # TPU v6e
    '0x006f',
    # Testing only
    '0x0056',
    '0x0062',
]


class TpuEnv(TypedDict):
  ACCELERATOR_TYPE: str
  TPU_PROCESS_BOUNDS: str
  TPU_CHIPS_PER_HOST_BOUNDS: str
  WORKER_ID: int


class MeshShape(NamedTuple):
  """Represents a TPU mesh shape (e.g. '2,2,1' or '1,1,1')"""
  x: int
  y: int
  z: int

  @classmethod
  def from_string(cls, mesh: str):
    dims = tuple(int(d) for d in mesh.split(','))
    if len(dims) != 3:
      raise ValueError("Mesh shape '{}' should be length 3".format(mesh))

    return MeshShape(*dims)

  @property
  def size(self) -> int:
    return functools.reduce(operator.mul, self)

  def __mul__(self, other):
    return MeshShape(*(d1 * d2 for d1, d2 in zip(self, other)))


def _get_metadata(key: str) -> str:
  path = os.path.join(_GCE_METADATA_ROOT_URL, 'instance/attributes', key)
  resp = requests.get(path, headers={'Metadata-Flavor': 'Google'})
  resp.raise_for_status()

  return resp.text


def process_bounds_size() -> Optional[int]:
  """Returns number of processes across all TPU hosts, or None if unknown."""
  process_bounds = xu.getenv_as(xenv.TPU_PROCESS_BOUNDS, str)
  return MeshShape.from_string(process_bounds).size if process_bounds else None


def num_available_chips() -> int:
  """Returns the number of TPU chips attached through PCI."""
  num_chips = 0
  for vendor_path in glob.glob('/sys/bus/pci/devices/*/vendor'):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_path = os.path.join(os.path.dirname(vendor_path), 'device')
    device_id = pathlib.Path(device_path).read_text().strip()
    if device_id in _TPU_PCI_DEVICE_IDS:
      num_chips += 1

  return num_chips


def num_logical_cores_per_chip() -> int:
  """Returns number of XLA TPU devices per physical chip on the current host."""
  return 2 if version() <= 3 else 1


def num_available_devices() -> int:
  """Returns number of XLA TPU devices on the current host.

  Note: this does not initialize the computation client and is safe to call
  before `xmp.spawn`.
  """
  return num_available_chips() * num_logical_cores_per_chip()


def num_expected_global_devices() -> int:
  """Returns the number of expected runtime devices in this TPU slice.

  May differ from the actual number of runtime devices if TPU topology settings
  are changed.
  """
  return num_available_devices() * num_tpu_workers()


def num_local_processes() -> int:
  """Returns number of processes to create on this host."""
  local_chips = num_available_chips()
  total_processes = process_bounds_size()
  # Don't create more processes than local chips
  return local_chips if not total_processes else min(local_chips,
                                                     total_processes)


def task_id() -> Optional[int]:
  """Returns index of this process within all TPU worker processes, if any."""
  return xu.getenv_as(xenv.CLOUD_TPU_TASK_ID, int)


def worker_id() -> int:
  """Returns the ID of the current TPU worker."""
  env = get_tpu_env()
  return int(env[xenv.WORKER_ID])


def _using_env_vars() -> bool:
  return xu.getenv_as(xenv.TPU_SKIP_MDS_QUERY, str, False)


def build_tpu_env_from_vars() -> TpuEnv:
  metadata = dict()
  metadata[xenv.ACCELERATOR_TYPE] = xu.getenv_as(xenv.TPU_ACCELERATOR_TYPE, str)
  metadata[xenv.TPU_PROCESS_BOUNDS] = xu.getenv_as(
      xenv.TPU_PROCESS_BOUNDS, str, xu.getenv_as(xenv.TPU_HOST_BOUNDS, str))
  metadata[xenv.TPU_CHIPS_PER_PROCESS_BOUNDS] = xu.getenv_as(
      xenv.TPU_CHIPS_PER_PROCESS_BOUNDS, str,
      xu.getenv_as(xenv.TPU_CHIPS_PER_HOST_BOUNDS, str))
  metadata[xenv.WORKER_ID] = xu.getenv_as(xenv.CLOUD_TPU_TASK_ID, str,
                                          xu.getenv_as(xenv.TPU_WORKER_ID, str))
  return metadata


def get_tpu_env() -> TpuEnv:
  """Fetches and parses `tpu-env` metadata field."""
  if _using_env_vars():
    return build_tpu_env_from_vars()
  metadata = _get_metadata('tpu-env')
  return yaml.load(metadata, yaml.Loader)


def version() -> int:
  try:
    env = get_tpu_env()
  except requests.HTTPError as e:
    raise EnvironmentError('Failed to get TPU metadata') from e

  match = re.match(r'^v(\d)([A-Za-z]?){7}-(\d+)$', env[xenv.ACCELERATOR_TYPE])
  return int(match.groups()[0])


def get_worker_ips() -> List[str]:
  """Returns ordered list of TPU worker IPs from TPU metadata."""
  if _using_env_vars():
    hostnames_string = xu.getenv_as(xenv.TPU_WORKER_HOSTNAMES, str, '')
    # String has the format 'host-name-1,host-name-2,...,host-name-n'
    hostnames = hostnames_string.split(',')
  else:
    hostnames_string = _get_metadata('worker-network-endpoints')
    # Workers have format 'hostname:uid:ip,hostname:uid:ip,...'
    workers = hostnames_string.split(',')
    hostnames = [worker.split(':')[2] for worker in workers]
  return hostnames if len(hostnames) > 1 else ['localhost']


def num_tpu_workers() -> int:
  """Returns the number of configured TPU workers."""
  return len(get_worker_ips())


def configure_one_chip_topology() -> None:
  """Configures TPU topology environment variables for one process and chip.

  Must be run before using any XLA devices.
  """
  os.environ.setdefault(xenv.TPU_VISIBLE_CHIPS, '0')
  os.environ.setdefault(xenv.TPU_CHIPS_PER_PROCESS_BOUNDS, '1,1,1')
  os.environ.setdefault(xenv.TPU_PROCESS_BOUNDS, '1,1,1')


def configure_topology(local_rank: int,
                       local_world_size: int,
                       base_port: int = 8476) -> None:
  """Configures TPU topology environment variables based on TPU metadata.

  Must be run before using any XLA devices.

  Args:
    local_rank: rank of this process within this host.
    local_world_size: number of processes on this host.
    base_port: starting port for TPU clients on each host. Ports in the range
      [base_port, base_port + local_world_size) must be free on each host.
    mesh_port: port to use for rendezvous operations. Must be free in process 0.
  """
  tpu_env = get_tpu_env()

  accelerator_type = tpu_env[xenv.ACCELERATOR_TYPE]
  if version() >= 4:
    # Process bounds with 4 chips per process
    default_process_bounds = MeshShape.from_string(
        tpu_env[xenv.TPU_PROCESS_BOUNDS])
    chips_per_process = MeshShape.from_string(
        tpu_env[xenv.TPU_CHIPS_PER_PROCESS_BOUNDS])
  else:
    # TODO: merge with TPU v4 case when bounds are added to metadata
    default_process_bounds = MeshShape.from_string(
        _ACCELERATOR_TYPE_TO_HOST_BOUNDS[accelerator_type])
    chips_per_process = MeshShape.from_string('2,2,1')

  # Process bounds with 1 chip per process
  process_bounds = default_process_bounds * chips_per_process

  os.environ.setdefault(xenv.TPU_CHIPS_PER_PROCESS_BOUNDS, '1,1,1')
  os.environ.setdefault(xenv.TPU_PROCESS_BOUNDS,
                        ','.join(str(dim) for dim in process_bounds))

  # Assume each TPU has the same number of local processes with the same ports
  worker_id = int(tpu_env[xenv.WORKER_ID])
  os.environ.setdefault(xenv.CLOUD_TPU_TASK_ID,
                        str(worker_id * local_world_size + local_rank))

  worker_ips = get_worker_ips()

  ports = list(range(base_port, base_port + local_world_size))
  process_endpoints = [
      ','.join(f'{ip}:{port}' for port in ports) for ip in worker_ips
  ]
  os.environ.setdefault(xenv.TPU_PROCESS_ADDRESSES, ','.join(process_endpoints))

  os.environ.setdefault(xenv.TPU_VISIBLE_CHIPS, str(local_rank))
  os.environ.setdefault(xenv.TPU_PROCESS_PORT, str(ports[local_rank]))


def discover_master_worker_ip(use_localhost: bool = True) -> str:
  """Find the IP of the master TPU host.

  In multiprocess, this is the host with TPU:0.
  In SPMD mode, this is the host running process 0.

  TPU device IDs are nondeterministic and independent from Cloud TPU worker IDs.

  Args:
    use_localhost: if there is only one TPU host, return 'localhost` instead
      of that host's internal IP.
  """
  import torch_xla.runtime as xr
  worker_ips = get_worker_ips()
  if len(worker_ips) == 1:
    return 'localhost'

  tpu_env = get_tpu_env()
  current_worker_id = int(tpu_env[xenv.WORKER_ID])
  if xr.is_spmd():
    return _spmd_find_master_ip(worker_ips[current_worker_id])

  t = torch.tensor([current_worker_id], device=xm.xla_device())
  xm.collective_broadcast([t])
  xm.mark_step()

  master_worker_id = int(t.cpu())
  return worker_ips[master_worker_id]


def _spmd_find_master_ip(current_worker_hostname: str) -> str:
  import torch_xla.runtime as xr
  import torch_xla.distributed.spmd as xs
  from_cpu_shards = torch_xla._XLAC._global_tensor_from_cpu_shards
  # Translate the hostname to an IP address, e.g. for TPUs on GKE.
  current_worker_ip = socket.gethostbyname(current_worker_hostname)
  ip_int = int(ip_address(current_worker_ip))
  n_dev = xr.global_runtime_device_count()
  local_ndev = len(torch_xla._XLAC._xla_get_runtime_devices())
  # Create a global (n_dev x 2) tensor containing all process indices and IPs,
  # and find the process 0 IP as the master IP.
  shard = torch.LongTensor([[xr.process_index(), ip_int]])
  op_sharding = xs.Mesh(range(n_dev), (n_dev, 1)).get_op_sharding((0, 1))
  global_tensor = from_cpu_shards([shard] * local_ndev, op_sharding).cpu()
  # Process 0 may not control device 0, so we must do a linear search.
  for proc, ip in global_tensor.tolist():
    if proc == 0:
      return str(ip_address(ip))
  raise RuntimeError('Could not find IP of host running process 0')


class TpuPlugin(plugins.DevicePlugin):

  def library_path(self):
    libtpu_path = os.getenv('TPU_LIBRARY_PATH') or os.getenv(
        'PTXLA_TPU_LIBRARY_PATH')
    if not libtpu_path:
      raise EnvironmentError('libtpu not found')

    return libtpu_path

  def host_index(self):
    return worker_id()

  def configure_single_process(self):
    return configure_one_chip_topology()

  def configure_multiprocess(self, local_rank, local_world_size):
    return configure_topology(local_rank, local_world_size)

  def physical_chip_count(self):
    # HACK: We may reduce the number of processes we spawn depending on TPU
    # topology settings
    return num_local_processes()

  def client_create_options(self):
    return {
        'max_inflight_computations':
            xu.getenv_as('XLA_TPU_MAX_INFLIGHT_COMPUTATIONS', int, 32),
        'ml_framework_name':
            'PyTorch/XLA',
        'ml_framework_version':
            __version__
    }
