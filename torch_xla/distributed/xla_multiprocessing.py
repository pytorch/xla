from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os
import re
import socket
import sys
import torch.multiprocessing
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import traceback

PreForkConfig = collections.namedtuple('PreForkConfig', 'dev_kind num_devices')
WorkerConfigEntry = collections.namedtuple('WorkerConfigEntry',
                                           'worker_name ordinal host_port')

_LOCAL_WORKER = 'localservice'
_CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'


def _get_free_tcp_ports(n=1):
  ports = []
  for _ in range(0, n):
    with contextlib.closing(socket.socket(socket.AF_INET,
                                          socket.SOCK_STREAM)) as s:
      s.bind(('', 0))
      ports.append(s.getsockname()[1])
  return ports


def _is_xla_config():
  for env in [xenv.TPU_CONFIG, xenv.LOCAL_WORKER, xenv.GPU_NUM_DEVICES]:
    if os.environ.get(env, None) is not None:
      return True
  return False


def _get_world_size():
  # We cannot use the xla_model.py API here, as the features used in that module
  # needs the setup provided by this one.
  return int(os.environ.get(xenv.WORLD_SIZE, '1'))


def _create_gpu_devices(num_gpus):
  devices = []
  for h in range(0, _get_world_size()):
    for i in range(0, num_gpus):
      gindex = h * num_gpus + i
      # We use CUDA_VISIBLE_DEVICES to limit the set of CUDA devices per process
      # to 1, and its device index is always 0. We use the task to disambiguate
      # TF devices.
      tfdevice = '/job:{}/replica:0/task:{}/device:XLA_GPU:0'.format(
          _LOCAL_WORKER, gindex)
      devices.append('GPU:{};{}'.format(gindex, tfdevice))
  os.environ[xenv.DEVICE_MAP] = '|'.join(devices)


def _parse_workers_config(config):
  # XRT_WORKERS='worker:0;ismz9:25822'
  workers = collections.OrderedDict()
  for worker in config.split('|'):
    m = re.match(r'(\w+):(\d+);((grpc://)?[\w.]+:\d+)', worker)
    if not m:
      raise ValueError('Bad worker syntax: {}'.format(worker))
    workers['{}:{}'.format(m.group(1), m.group(2))] = WorkerConfigEntry(
        worker_name=m.group(1), ordinal=int(m.group(2)), host_port=m.group(3))
  return workers


def _parse_tpu_config(config):
  # XRT_TPU_CONFIG='tpu_worker;0;ismz9:25822'
  workers = collections.OrderedDict()
  for worker in config.split('|'):
    m = re.match(r'(\w+);(\d+);([\w.]+:\d+)', worker)
    if not m:
      raise ValueError('Bad worker syntax: {}'.format(worker))
    workers['{}:{}'.format(m.group(1), m.group(2))] = WorkerConfigEntry(
        worker_name=m.group(1), ordinal=int(m.group(2)), host_port=m.group(3))
  return workers


def _get_devices_per_worker():
  if os.environ.get(xenv.TPU_CONFIG, None) is not None:
    return int(os.environ.get(xenv.TPU_NUM_DEVICES, '8')), 'TPU'
  num_gpus = os.environ.get(xenv.GPU_NUM_DEVICES, None)
  if num_gpus is not None:
    return int(num_gpus), 'GPU'
  raise RuntimeError('Missing TPU or GPU configuration')


def _get_multiprocessing_device():
  return os.environ.get(xenv.MP_DEVICE, None)


def _get_local_worker_index():
  worker = os.environ.get(xenv.LOCAL_WORKER, None)
  if worker is None:
    return 0
  m = re.match(r'(\w+):(\d+)', worker)
  if not m:
    raise ValueError('Bad worker syntax: {}'.format(worker))
  return int(m.group(2))


def _local_index_to_global(index, num_devices):
  return _get_local_worker_index() * num_devices + index


def _setup_world_size(num_devices):
  # We cannot call into xla_model code at this point, as we do not know whether
  # the called code would trigger XLA library initializations (which we must
  # not do at this point). So we avoid calling into xm.xrt_world_size().
  world_size = _get_world_size() * num_devices
  os.environ[xenv.WORLD_SIZE] = str(world_size)


def _setup_workers(num_devices):
  world_size = _get_world_size()
  workers_env = os.environ.get(xenv.WORKERS, None)
  workers = []
  if workers_env is not None:
    wcfg = _parse_workers_config(workers_env)
    assert world_size == len(
        wcfg), 'World size ({}) must match the configured workers ({})'.format(
            world_size, len(wcfg))
    for h, worker in enumerate(wcfg):
      m = re.match(r'(.*):(\d+)$', worker.host_port)
      if not m:
        raise RuntimeError('Bad worker HOST:PORT format: {}'.format(
            worker.host_port))
      for i in range(0, num_gpus):
        gindex = h * num_gpus + i
        workers.append('{}:{};grpc://{}:{}'.format(worker.worker_name, gindex,
                                                   m.group(1),
                                                   int(m.group(2)) + i))
  else:
    assert world_size == 1, ('Cannot use more than one host without {} '
                             'configuration: {}').format(
                                 xenv.WORKERS, world_size)
    ports = _get_free_tcp_ports(num_devices)
    host = socket.getfqdn()
    for wid in range(0, num_devices):
      workers.append('{}:{};grpc://{}:{}'.format(_LOCAL_WORKER, wid, host,
                                                 ports[wid]))
  os.environ[xenv.WORKERS] = '|'.join(workers)


def _pre_fork_setup(num_devices):
  dev_count, dev_kind = _get_devices_per_worker()
  if num_devices is None:
    num_devices = dev_count
  elif num_devices not in [1, dev_count]:
    raise ValueError(
        'The number of devices must be either 1 or {}, got {} instead'.format(
            dev_count, num_devices))
  if num_devices > 1 and not os.environ.get(xenv.SERVICE_ADDRESS, None):
    # In multi-processing mode, even if there is only one XLA host, we still
    # bring up the mesh service.
    os.environ[xenv.SERVICE_ADDRESS] = '{}:{}'.format(socket.getfqdn(),
                                                      _get_free_tcp_ports()[0])
  if dev_kind == 'GPU':
    _setup_workers(num_devices)
    _create_gpu_devices(num_devices)
  return PreForkConfig(dev_kind=dev_kind, num_devices=num_devices)


def _setup_gpu_worker(index, gindex, pf_cfg):
  os.environ[xenv.MP_DEVICE] = 'GPU:{}'.format(gindex)
  os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(_LOCAL_WORKER, gindex)
  # Every process is restricted to 1 GPU device, which in such process will be
  # named XLA_GPU:0.
  os.environ[_CUDA_VISIBLE_DEVICES] = str(index)


def _setup_tpu_worker(index, gindex, pf_cfg, tpu_env_config):
  os.environ[xenv.MP_DEVICE] = 'TPU:{}'.format(gindex)
  if xenv.LOCAL_WORKER not in os.environ:
    # The local worker can be missing for a 1 TPU host setup. Make sure we
    # always have one.
    tpu_config = _parse_tpu_config(tpu_env_config)
    worker = list(tpu_config.values())[0]
    os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(worker.worker_name,
                                                   worker.ordinal)
  if gindex > 0 and xenv.TPU_CONFIG in os.environ:
    # In multi-processing mode, only the process handling the first device of
    # the master worker, will do TPU mesh initialization.
    del os.environ[xenv.TPU_CONFIG]


def _prepare_env_for_index(index, pf_cfg):
  _setup_world_size(pf_cfg.num_devices)
  gindex = _local_index_to_global(index, pf_cfg.num_devices)
  os.environ[xenv.ORDINAL] = str(gindex)
  os.environ[xenv.LOCAL_ORDINAL] = str(index)

  if pf_cfg.dev_kind == 'TPU':
    _setup_tpu_worker(index, gindex, pf_cfg, os.environ[xenv.TPU_CONFIG])
  elif pf_cfg.dev_kind == 'GPU':
    _setup_gpu_worker(index, gindex, pf_cfg)
  return gindex


def _setup_replication():
  # At this point xla_model.py APIs are allowed as the setup is already
  # completed.
  if xm.xrt_world_size() > 1:
    device = xm.xla_device()
    xm.set_replication(str(device), [str(device)])


def _start_fn(index, pf_cfg, fn, args):
  gindex = _prepare_env_for_index(index, pf_cfg)
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


def _run_direct(fn, args, nprocs, join, daemon, start_method):
  nprocs = nprocs or 1
  if nprocs == 1 and join:
    fn(0, *args)
  else:
    return torch.multiprocessing.spawn(
        fn, args=args, nprocs=nprocs, join=join, daemon=daemon)


def spawn(fn,
          args=(),
          nprocs=None,
          join=True,
          daemon=False,
          start_method='spawn'):
  """Enables multi processing based replication.

  Args:
    fn (callable): The function to be called for each device which takes part of
      the replication. The function will be called with a first argument being
      the global index of the process within the replication, followed by the
      arguments passed in `args`.
    args (tuple): The arguments for `fn`.
      Default: Empty tuple
    nprocs (int): The number of processes/devices for the replication. At the
      moment, if specified, can be either 1 or the maximum number of devices.
    join (bool): Whether the call should block waiting for the completion of the
      processes which have being spawned.
      Default: True
    daemon (bool): Whether the processes being spawned should have the `daemon`
      flag set (see Python multi-processing API).
      Default: False
    start_method (string): The Python `multiprocessing` process creation mathod.
      Default: `spawn`

  Returns:
    The same object returned by the `torch.multiprocessing.spawn` API. If
    `nprocs` is 1 the `fn` function will be called directly, and the API will
    not return.
  """
  if not _is_xla_config():
    # If this is not an XLA setup, jump to normal multi-processing.
    return _run_direct(fn, args, nprocs, join, daemon, start_method)

  pf_cfg = _pre_fork_setup(nprocs)
  if pf_cfg.num_devices == 1:
    _start_fn(0, pf_cfg, fn, args)
  else:
    return torch.multiprocessing.start_processes(
        _start_fn,
        args=(pf_cfg, fn, args),
        nprocs=pf_cfg.num_devices,
        join=join,
        daemon=daemon,
        start_method=start_method)


class MpModelWrapper(object):
  """Wraps a model to minimize host memory usage when `fork` method is used.

  This class should be used together with the `spawn(..., start_method='fork')`
  API to minimize the use of host memory.
  Instead of creating models on each multiprocessing process, hence replicating
  the model's initial host memory, the model is created once at global scope,
  and then moved into each device inside the `spawn()` target function.
  Example::

    WRAPPED_MODEL = xmp.MpModelWrapper(MyNetwork())

    def _mp_fn(index, ...):
      device = xm.xla_device()
      model = WRAPPED_MODEL.to(device)
      ...

    xmp.spwan(_mp_fn, ..., start_method='fork')

  This method has two advantages. First if uses only one copy of the memory
  pages to host the original model weights, and second it serializes the move
  of the wrapped model into each device, by lowering the load onto the system
  memory during the process.
  """

  def __init__(self, model):
    """Creates a new `MpModelWrapper` object.

    Args:
      model (torch.nn.Module): The model to be wrapped. Should be on PyTorch CPU
        device (which is the default when creating new models).
    """
    self._model = model
    self._lock = torch.multiprocessing.Lock()

  def to(self, device):
    """Retrieves the model moved onto the specified device.

    Args:
      device (torch.device): The device where the model should be moved onto.
    Returns:
      The model on the specified device.
    """
    with self._lock:
      self._model.to(device)
    return self._model
