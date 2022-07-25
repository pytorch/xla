import cloud_tpu_client
import os
from typing import Optional, Iterable, Tuple
import numpy as np
import numpy.typing as npt
import requests
import yaml

import torch_xla.utils.utils as xu
import torch_xla.core.xla_env_vars as xenv

_GCE_METADATA_ROOT_URL = 'http://metadata.google.internal/computeMetadata/v1'

MeshShape = Tuple[int, int, int]

def _parse_mesh_shape(mesh: str) -> MeshShape:
  dims = tuple(int(d) for d in mesh.split(','))
  if len(dims) != 3:
    raise ValueError("Mesh shape '{}' should be length 3".format(mesh))

  return dims

def _multiple_mesh_shapes(mesh1: MeshShape, mesh2: MeshShape) -> MeshShape:
  return tuple(d1 * d2 for d1, d2 in zip(mesh1, mesh2))

def _get_metadata(key: str) -> str:
  path = os.path.join(_GCE_METADATA_ROOT_URL, 'instance/attributes', key)
  resp = requests.get(path, headers={'Metadata-Flavor': 'Google'})
  resp.raise_for_status()

  return resp.text

def task_id() -> Optional[int]:
  return xu.getenv_as(xenv.CLOUD_TPU_TASK_ID, int)

def get_tpu_env():
  metadata = _get_metadata('tpu-env')

  return yaml.load(metadata, yaml.Loader)

def configure_topology(local_rank: int, local_world_size: int, base_port: int = 8476):
  tpu_env = get_tpu_env()

  # Process bounds with 4 chips per process
  default_process_bounds = _parse_mesh_shape(tpu_env[xenv.TPU_PROCESS_BOUNDS])
  chips_per_process = _parse_mesh_shape(tpu_env[xenv.TPU_CHIPS_PER_PROCESS_BOUNDS])

  # Process bounds with 1 chip per process
  process_bounds = _multiple_mesh_shapes(default_process_bounds, chips_per_process)

  os.environ.setdefault(xenv.TPU_CHIPS_PER_PROCESS_BOUNDS, '1,1,1')
  os.environ.setdefault(xenv.TPU_PROCESS_BOUNDS, ','.join(str(dim) for dim in process_bounds))

  # Assume each TPU has the same number of local processes with the same ports
  worker_id = int(tpu_env['WORKER_ID'])
  os.environ.setdefault(xenv.CLOUD_TPU_TASK_ID, str(worker_id * local_world_size + local_rank))

  client = cloud_tpu_client.Client(tpu=tpu_env['NODE_ID'])
  host_ips = [e['ipAddress'] for e in client.network_endpoints()]

  ports = list(range(base_port, base_port + local_world_size))
  process_endpoints = [','.join(f'{ip}:{port}' for port in ports) for ip in host_ips]
  os.environ.setdefault(xenv.TPU_PROCESS_ADDRESSES, ','.join(process_endpoints))

  os.environ.setdefault(xenv.TPU_VISIBLE_DEVICES, str(local_rank))
  os.environ.setdefault(xenv.TPU_PROCESS_PORT, str(ports[local_rank]))
