from collections import defaultdict
import functools
import operator
import os
import re
from typing import Dict, NamedTuple, Optional, List, Tuple
import requests
import yaml

import torch
import torch_xla.utils.utils as xu
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm

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


def process_bounds_size(default: int = 1) -> int:
  """Returns number of processes across all TPU hosts."""
  process_bounds = xu.getenv_as(xenv.TPU_PROCESS_BOUNDS, str)
  return MeshShape.from_string(
      process_bounds).size if process_bounds else default


def num_local_processes(local_chips: int = 4) -> int:
  """Returns number of processes to create on this host."""
  # Don't create more processes than local chips
  return min(local_chips, process_bounds_size(default=local_chips))


def task_id() -> Optional[int]:
  """Returns index of this process within all TPU worker processes, if any."""
  return xu.getenv_as(xenv.CLOUD_TPU_TASK_ID, int)

def build_tpu_env_from_vars() -> Dict[str, str]:
  metadata = defaultdict(str)
  metadata[xenv.ACCELERATOR_TYPE] = xu.getenv_as(xenv.TPU_ACCELERATOR_TYPE, str)
  metadata[xenv.TPU_PROCESS_BOUNDS] = xu.getenv_as(xenv.TPU_PROCESS_BOUNDS, str, xu.getenv_as(xenv.TPU_HOST_BOUNDS, str))
  metadata[xenv.TPU_CHIPS_PER_PROCESS_BOUNDS] = xu.getenv_as(xenv.TPU_CHIPS_PER_PROCESS_BOUNDS, str, xu.getenv_as(xenv.TPU_CHIPS_PER_HOST_BOUNDS, str))
  metadata[xenv.WORKER_ID] = xu.getenv_as(xenv.CLOUD_TPU_TASK_ID, str, xu.getenv_as(xenv.TPU_WORKER_ID, str))
  return metadata


def get_tpu_env() -> Dict[str, str]:
  """Fetches and parses `tpu-env` metadata field."""
  metadata = build_tpu_env_from_vars()
  if metadata[xenv.ACCELERATOR_TYPE] is '':
      metadata = _get_metadata('tpu-env')
      return yaml.load(metadata, yaml.Loader)
  return metadata


def version() -> int:
  try:
    env = get_tpu_env()
  except requests.HTTPError as e:
    raise EnvironmentError('Failed to get TPU metadata') from e

  match = re.match(r'^v(\d)-(\d+)$', env[xenv.ACCELERATOR_TYPE])
  return int(match.groups()[0])


def get_worker_ips() -> List[str]:
  """Returns ordered list of TPU worker IPs from TPU metadata."""
  metadata = xu.getenv_as(xenv.TPU_WORKER_HOSTNAMES, str, '')
  if metadata is '':
      metadata = _get_metadata('worker-network-endpoints')
  # Workers have format 'hostname:uid:ip,hostname:uid:ip,...'
  workers = metadata.split(',')
  ips = [worker.split(':')[2] for worker in workers]

  return ips if len(ips) > 1 else ['localhost']


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
  if tpu_env[xenv.ACCELERATOR_TYPE].startswith('v4'):  
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
  """Find the IP of the TPU host with TPU:0.

  TPU device IDs are nondeterministic and independent from Cloud TPU worker IDs.

  Args:
    use_localhost: if there is only one TPU host, return 'localhost` instead
      of that host's internal IP.
  """
  worker_ips = get_worker_ips()
  if len(worker_ips) == 1:
    return 'localhost'

  tpu_env = get_tpu_env()
  current_worker_id = int(tpu_env[xenv.WORKER_ID])
  t = torch.tensor([current_worker_id], device=xm.xla_device())
  xm.collective_broadcast([t])
  xm.mark_step()

  master_worker_id = int(t.cpu())
  return worker_ips[master_worker_id]
