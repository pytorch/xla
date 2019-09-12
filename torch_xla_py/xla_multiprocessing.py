from __future__ import division
from __future__ import print_function

import contextlib
import os
import socket
import sys
import torch.multiprocessing
import torch_xla
import torch_xla_py.xla_env_vars as xenv
import torch_xla_py.xla_model as xm
import traceback


def _find_free_tcp_port():
  with contextlib.closing(socket.socket(socket.AF_INET,
                                        socket.SOCK_STREAM)) as s:
    s.bind(('', 0))
    return s.getsockname()[1]


def _is_tpu_config():
  for env in [xenv.TPU_CONFIG, xenv.LOCAL_WORKER]:
    if os.environ.get(env, None) is not None:
      return True
  return False


def _parse_tpu_config(config):
  # XRT_TPU_CONFIG='tpu_worker;0;ismz9:25822'
  parsed = []
  for worker in config.split('|'):
    parts = worker.split(';')
    if len(parts) != 3:
      raise ValueError('Bad worker syntax: {}'.format(worker))
    parsed.append((parts[0], int(parts[1]), parts[2]))
  return parsed


def _get_devices_per_worker():
  return int(os.environ.get(xenv.TPU_NUM_DEVICES, '8'))


def _get_multiprocessing_device():
  return os.environ.get(xenv.MP_DEVICE, None)


def _get_local_worker_index():
  worker = os.environ.get(xenv.LOCAL_WORKER, None)
  return int(worker.split(':')[1]) if worker is not None else 0


def _local_index_to_global(index):
  return _get_local_worker_index() * _get_devices_per_worker() + index


def _pre_fork_setup(num_devices):
  if num_devices is None:
    num_devices = _get_devices_per_worker()
  elif num_devices not in [1, _get_devices_per_worker()]:
    raise ValueError(
        'The number of devices must be either 1 or {}, got {} instead'.format(
            _get_devices_per_worker(), num_devices))
  # We cannot call into xla_model code at this point, as we do not know whether
  # the called code would trigger XLA library initializations (which we must
  # not do at this point). So we avoid calling into xm.xrt_world_size().
  world_size = int(os.environ.get(xenv.WORLD_SIZE, '1')) * num_devices
  os.environ[xenv.WORLD_SIZE] = str(world_size)
  if not os.environ.get(xenv.SERVICE_ADDRESS, None):
    # In multi-processing mode, even if there is only one TPU host, we still
    # bring up the mesh service.
    os.environ[xenv.SERVICE_ADDRESS] = 'localhost:{}'.format(
        _find_free_tcp_port())
  return num_devices


def _prepare_env_for_index(index):
  gindex = _local_index_to_global(index)
  os.environ[xenv.MP_DEVICE] = 'TPU:{}'.format(gindex)
  os.environ[xenv.ORDINAL] = str(gindex)
  if xenv.LOCAL_WORKER not in os.environ:
    # The local worker can be missing for a 1 TPU host setup. Make sure we
    # always have one.
    tpu_config = _parse_tpu_config(os.environ[xenv.TPU_CONFIG])
    worker = tpu_config[0]
    os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(worker[0], worker[1])
  if gindex > 0 and xenv.TPU_CONFIG in os.environ:
    # In multi-processing mode, only the process handling the first device of
    # the master worker, will do TPU mesh initialization.
    del os.environ[xenv.TPU_CONFIG]
  return gindex


def _setup_replication():
  if xm.xrt_world_size() > 1:
    device = xm.xla_device()
    xm.set_replication(str(device), [str(device)])


def _start_fn(index, fn, args):
  gindex = _prepare_env_for_index(index)
  # Calling _setup_replication() will trigger XLA library initialization, so the
  # environment must be fully setup before doing so.
  _setup_replication()
  exit_code = 0
  try:
    fn(gindex, *args)
  except Exception as e:
    print(
        'Exception in device={}: {}'.format(_get_multiprocessing_device(),
                                            str(e)),
        file=sys.stderr)
    traceback.print_exc(limit=16, file=sys.stderr)
    exit_code = 17
  sys.exit(exit_code)


def spawn(fn, args=(), nprocs=None, join=True, daemon=False):
  if not _is_tpu_config():
    # If this is not an TPU setup, jump to normal multi-processing.
    nprocs = nprocs or 1
    return torch.multiprocessing.spawn(
        fn, args=args, nprocs=nprocs, join=join, daemon=daemon)

  nprocs = _pre_fork_setup(nprocs)
  return torch.multiprocessing.spawn(
      _start_fn, args=(fn, args), nprocs=nprocs, join=join, daemon=daemon)
