import os
import re
import socket
import subprocess
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
from torch_xla.utils.utils import get_free_tcp_ports

XRT_SERVER_REGEX = 'torch_xla.distributed._xrt_run_server'
_TCP_STORE = None
_INIT_XRT_ALREADY_CALLED = False


def _create_devices(dev_kind, world_size):
  # Create global XLA devices. Adapted from xmp.spawn() to function across nodes
  devices = []
  dev_type = 'GPU'

  for gindex in range(0, world_size):
    tfdevice = f'{dev_type}:{gindex};/job:localservice/replica:0/task:{gindex}/device:XLA_{dev_type}:0'
    devices.append(tfdevice)
  os.environ[xenv.DEVICE_MAP] = '|'.join(devices)


def _setup_workers(world_size, rank, local_world_size, local_rank):
  # Set up workers across nodes. xmp.spawn() does this locally by figuring out free ports on the node
  # We do this globally by doing an allgather of locally obtained free socket addresses
  # Note that this follows the original scheme, in the new scheme only one address per node needs exchange
  host = socket.gethostname()
  if local_rank == 0:
    ports = [str(i) for i in get_free_tcp_ports(local_world_size)]
    _TCP_STORE.set(host, ' '.join(ports))
  else:
    ports_str = _TCP_STORE.get(host).decode('UTF-8')
    ports = list(ports_str.split(' '))

  my_worker = '{}:{};grpc://{}:{}'.format('localservice', rank, host,
                                          ports[local_rank])
  all_workers = []
  for i in range(0, world_size):
    if rank == i:
      _TCP_STORE.set(f'worker:{i}', my_worker)
      all_workers.append(my_worker)
    else:
      worker = _TCP_STORE.get(f'worker:{i}').decode('UTF-8')
      all_workers.append(worker)
  os.environ['XRT_WORKERS'] = '|'.join(all_workers)


def _get_address_from_store(key, rank):
  if rank == 0:
    port = get_free_tcp_ports()[0]
    host = socket.getfqdn()
    service_addr = '{}:{}'.format(host, port)
    _TCP_STORE.set(key, service_addr)
  else:
    service_addr = _TCP_STORE.get(key).decode('UTF-8')

  return service_addr


def _set_mesh_config(rank):
  address = _get_address_from_store('xrt_mesh_config', rank)
  if not os.environ.get(xenv.SERVICE_ADDRESS, None):
    os.environ[xenv.SERVICE_ADDRESS] = address
  if not os.environ.get("TPU_MESH_CONTROLLER_ADDRESS", None):
    address = _get_address_from_store('tpu_mesh_config', rank)
    _, port = address.split(":")
    os.environ["TPU_MESH_CONTROLLER_ADDRESS"] = address
    os.environ["TPU_MESH_CONTROLLER_PORT"] = port


def _set_tpu_xrt_envs(local_rank, rank, group_rank, local_world_size,
                      world_size):
  total_nodes = world_size // local_world_size

  xrt_tpu_config = []
  tpu_config_port = None
  for i in range(total_nodes):
    key = f'worker_{i}_address'
    if group_rank == i and local_rank == 0:
      tpu_config_port = get_free_tcp_ports()[0]
      host = socket.getfqdn()
      address = '{}:{}'.format(host, tpu_config_port)
      _TCP_STORE.set(key, address)
    else:
      address = _TCP_STORE.get(key).decode('UTF-8')
    if total_nodes == 1:
      xrt_tpu_config.append(f'localservice;{i};{address}')
    else:
      xrt_tpu_config.append(f'c_localservice;{i};{address}')

    if rank == 0:
      os.environ[xenv.TPU_CONFIG] = '|'.join(xrt_tpu_config)
      os.environ[xenv.TPU_NUM_DEVICES] = str(local_world_size)

  os.environ[
      xenv.
      LOCAL_WORKER] = f'localservice:{group_rank}' if total_nodes == 1 else f'c_localservice:{group_rank}'
  os.environ[xenv.WORLD_SIZE] = str(world_size)
  os.environ[xenv.HOST_WORLD_SIZE] = str(total_nodes)
  os.environ[xenv.ORDINAL] = str(rank)
  os.environ[xenv.LOCAL_ORDINAL] = str(local_rank)
  os.environ[xenv.MP_DEVICE] = f'TPU:{rank}'
  if not os.environ.get('TF_GRPC_DEFAULT_OPTIONS', None):
    os.environ['TF_GRPC_DEFAULT_OPTIONS'] = (
        'grpc.keepalive_time_ms=60000,grpc.keepalive_timeout_ms=14400000,'
        'grpc.http2.max_pings_without_data=0,grpc.http2.min_ping_interval_without_data_ms=300000'
    )
  # We don't want torch_xla to start the local server internally.
  # We are starting the xrt server by ourselves
  os.environ['XRT_START_LOCAL_SERVER'] = '0'

  return tpu_config_port


def _set_neuron_envs(rank, world_size, local_world_size):
  os.environ["NEURON_USE_LOAD_COLLECTIVES"] = '1'
  os.environ['NEURON_GLOBAL_DEVICE_ID'] = str(rank)
  os.environ['NEURON_GLOBAL_DEVICE_COUNT'] = str(world_size)
  if not os.environ.get('NEURON_RT_VISIBLE_CORES', None):
    os.environ['NEURON_RT_VISIBLE_CORES'] = ','.join(
        [str(i) for i in range(local_world_size)])


def _setup_nccl_service(dev_kind, rank):
  # Set up NCCL COMM ID required for NCCL communicator IDs
  address = _get_address_from_store('nccl_info', rank)
  if dev_kind == 'NEURON':
    os.environ['NEURON_RT_ROOT_COMM_ID'] = address
  elif dev_kind == 'GPU':
    os.environ['NEURON_RT_ROOT_COMM_ID'] = address
    os.environ['XRT_MESH_SERVICE_ADDRESS'] = address
  else:
    raise RuntimeError('NCCL service setup failed!')


def set_xrt_envs(world_size, rank, local_rank):
  # Set up all the XRT specific env variables, adapted from xmp.spawn()
  os.environ[xenv.WORLD_SIZE] = str(world_size)
  os.environ[xenv.ORDINAL] = str(rank)
  os.environ[xenv.LOCAL_ORDINAL] = str(local_rank)
  os.environ[xenv.LOCAL_WORKER] = 'localservice:' + str(rank)

  os.environ[xenv.MP_DEVICE] = f'GPU:{rank}'
  gpus_to_use = os.environ.get('CUDA_VISIBLE_DEVICES')
  if gpus_to_use is not None:
    # If gpu devices are set by a scheduling entity (eg. SLURM) we index into
    # comma separated string containing numbered gpu devies
    gpus_to_use_list = gpus_to_use.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use_list[local_rank]
  else:
    # If no explicit visible devices are provided, local_rank is used to identify
    # the gpu used by this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
