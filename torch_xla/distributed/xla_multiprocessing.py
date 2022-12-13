from __future__ import division
from __future__ import print_function

import collections
import os
import re
import socket
import sys
import torch.multiprocessing
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt
import torch_xla.utils.utils as xu
import traceback

PreForkConfig = collections.namedtuple('PreForkConfig', 'dev_kind num_devices')
WorkerConfigEntry = collections.namedtuple('WorkerConfigEntry',
                                           'worker_name ordinal host_port')

_LOCAL_WORKER = 'localservice'
_CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'


def _is_xla_config():
  for env in [
      xenv.TPU_CONFIG, xenv.LOCAL_WORKER, xenv.GPU_NUM_DEVICES,
      xenv.CPU_NUM_DEVICES
  ]:
    if os.environ.get(env, None) is not None:
      return True
  return False


# TODO: Some usages of this function are to caculate the number of hosts (a TPU concept),
# and some are to caculate the number of processes within a world (which can span multiple hosts).
# The latter should really be what this function is supposed to do. It's so confusing. We
# should improve it.
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
    m = re.match(r'(\w+):(\d+);((grpc://)?[a-zA-Z0-9_\-\.]+:\d+)', worker)
    if not m:
      raise ValueError('Bad worker syntax: {}'.format(worker))
    workers['{}:{}'.format(m.group(1), m.group(2))] = WorkerConfigEntry(
        worker_name=m.group(1), ordinal=int(m.group(2)), host_port=m.group(3))
  return workers


def _parse_tpu_config(config):
  # XRT_TPU_CONFIG='tpu_worker;0;ismz9:25822'
  workers = collections.OrderedDict()
  for worker in config.split('|'):
    m = re.match(r'(\w+);(\d+);([a-zA-Z0-9_\-\.]+:\d+)', worker)
    if not m:
      raise ValueError('Bad worker syntax: {}'.format(worker))
    workers['{}:{}'.format(m.group(1), m.group(2))] = WorkerConfigEntry(
        worker_name=m.group(1), ordinal=int(m.group(2)), host_port=m.group(3))
  return workers


def _get_devices_per_worker():
  num_tpus = os.environ.get(xenv.TPU_NUM_DEVICES, None)
  if os.environ.get(xenv.TPU_CONFIG, None) is not None or num_tpus is not None:
    return int(num_tpus or '8'), 'TPU'
  num_gpus = os.environ.get(xenv.GPU_NUM_DEVICES, None)
  if num_gpus is not None:
    return int(num_gpus), 'GPU'
  num_cpus = os.environ.get(xenv.CPU_NUM_DEVICES, None)
  if num_cpus is not None:
    return int(num_cpus), 'CPU'
  raise RuntimeError('Missing TPU or GPU configuration')


def _get_multiprocessing_device():
  return os.environ.get(xenv.MP_DEVICE, None)


def _get_local_worker_index():
  host_ordinal = os.environ.get(xenv.HOST_ORDINAL, None)
  if host_ordinal is not None:
    return int(host_ordinal)
  worker = os.environ.get(xenv.LOCAL_WORKER, None)
  if worker is None:
    return 0
  m = re.match(r'(\w+):(\d+)', worker)
  if not m:
    raise ValueError('Bad worker syntax: {}'.format(worker))
  return int(m.group(2))


def _local_index_to_global(index, num_devices):
  return _get_local_worker_index() * num_devices + index


def _setup_torch_distributed():
  import torch.distributed as dist

  ordinal = int(os.environ[xenv.HOST_ORDINAL])
  world_size = int(os.environ[xenv.HOST_WORLD_SIZE])
  method = os.environ.get(xenv.TORCH_DIST_METHOD, 'gloo')
  init_method = 'tcp://{}'.format(os.environ[xenv.TORCH_DIST_ROOT])
  dist.init_process_group(
      method, init_method=init_method, rank=ordinal, world_size=world_size)


def _setup_world_size(pf_cfg):
  # We cannot call into xla_model code at this point, as we do not know whether
  # the called code would trigger XLA library initializations (which we must
  # not do at this point). So we avoid calling into xm.xrt_world_size().
  host_world_size = _get_world_size()
  world_size = host_world_size * pf_cfg.num_devices
  os.environ[xenv.WORLD_SIZE] = str(world_size)
  if pf_cfg.dev_kind == 'CPU':
    # Since XLA CPU does not support across device reduces, and suport only one
    # device per process, we make each CPU device look like if it was a single
    # process host, and use torch.distributed for inter-host reductions (like in
    # the sea-of-devices case).
    os.environ[xenv.HOST_WORLD_SIZE] = str(world_size)
  else:
    os.environ[xenv.HOST_WORLD_SIZE] = str(host_world_size)


def _get_mp_device_ordinal(index, gindex):
  # If xenv.HOST_ORDINAL is set, we are in a sea-of-devices setup, where devices
  # are numbered locally within the single host (but the ordinal/rank is still
  # global).
  return index if xenv.HOST_ORDINAL in os.environ else gindex


# TODO: Consolidate this with _setup_gpu_worker.
def _setup_gpu_workers(num_devices):
  world_size = _get_world_size()
  workers_env = os.environ.get(xenv.WORKERS, None)
  workers = []
  # TODO: Is this path actually being used? This seems to support multi-host GPUs (is this a thing at all?).
  if workers_env is not None:
    wcfg = _parse_workers_config(workers_env)
    assert world_size == len(
        wcfg), 'World size ({}) must match the configured workers ({})'.format(
            world_size, len(wcfg))
    for key, worker in wcfg.items():
      _, ordinal = key.split(":")
      m = re.match(r'(.*):(\d+)$', worker.host_port)
      if not m:
        raise RuntimeError('Bad worker HOST:PORT format: {}'.format(
            worker.host_port))
      for i in range(0, num_devices):
        gindex = int(ordinal) * num_devices + i
        workers.append('{}:{};grpc://{}:{}'.format(worker.worker_name, gindex,
                                                   m.group(1),
                                                   int(m.group(2)) + i))
  else:
    assert world_size == 1, ('Cannot use more than one host without {} '
                             'configuration: {}').format(
                                 xenv.WORKERS, world_size)
    ports = xu.get_free_tcp_ports(num_devices)
    host = socket.getfqdn()
    for wid in range(0, num_devices):
      workers.append('{}:{};grpc://{}:{}'.format(_LOCAL_WORKER, wid, host,
                                                 ports[wid]))
  os.environ[xenv.WORKERS] = '|'.join(workers)


def _pre_fork_setup_torch_distributed():
  if not xenv.TORCH_DIST_ROOT in os.environ:
    os.environ[xenv.TORCH_DIST_ROOT] = '{}:{}'.format(
        socket.getfqdn(),
        xu.get_free_tcp_ports()[0])


def _pre_fork_cpu_setup(num_devices):
  if xenv.HOST_ORDINAL not in os.environ:
    # CPU multi-processing must use the host ordinal path, which enables the
    # torch.distributed reductions across single CPU cores. Since XLA CPU does
    # not support multiple devices within the same process, each XLA CPU device
    # is isolated within a single process, which is seen as "host" as well.
    os.environ[xenv.HOST_ORDINAL] = '0'


def _pre_fork_setup(num_devices):
  dev_count, dev_kind = _get_devices_per_worker()
  if num_devices is None:
    num_devices = dev_count
  elif num_devices not in [1, dev_count]:
    raise ValueError(
        'The number of devices must be either 1 or {}, got {} instead'.format(
            dev_count, num_devices))
  total_devices = _get_world_size() * num_devices
  if total_devices > 1 and not os.environ.get(xenv.SERVICE_ADDRESS, None):
    # In multi-processing mode, even if there is only one XLA host, we still
    # bring up the mesh service.
    os.environ[xenv.SERVICE_ADDRESS] = '{}:{}'.format(
        socket.getfqdn(),
        xu.get_free_tcp_ports()[0])
  if dev_kind == 'GPU':
    _setup_gpu_workers(num_devices)
    _create_gpu_devices(num_devices)
  elif dev_kind == 'CPU':
    _pre_fork_cpu_setup(num_devices)
  _pre_fork_setup_torch_distributed()
  return PreForkConfig(dev_kind=dev_kind, num_devices=num_devices)


def _setup_gpu_worker(index, gindex):
  os.environ[xenv.MP_DEVICE] = 'GPU:{}'.format(
      _get_mp_device_ordinal(index, gindex))
  os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(_LOCAL_WORKER, gindex)
  # Every process is restricted to 1 GPU device, which in such process will be
  # named XLA_GPU:0.
  os.environ[_CUDA_VISIBLE_DEVICES] = str(index)
  # We have expanded the GPU devices in the device map already, in
  # _create_gpu_devices(), so delete the key from the environment as it
  # otherwise triggers device generation again in computation_client.cc.
  os.environ.pop(xenv.GPU_NUM_DEVICES, None)


def _setup_cpu_worker(index, gindex):
  task_no = 0
  dev_index = _get_mp_device_ordinal(index, gindex)
  os.environ[xenv.MP_DEVICE] = 'CPU:{}'.format(dev_index)
  os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(_LOCAL_WORKER, task_no)
  os.environ[xenv.WORKERS] = '{}:{};grpc://localhost:{}'.format(
      _LOCAL_WORKER, task_no,
      xu.get_free_tcp_ports()[0])
  os.environ[
      xenv.
      DEVICE_MAP] = 'CPU:{};/job:{}/replica:0/task:{}/device:XLA_CPU:0'.format(
          dev_index, _LOCAL_WORKER, task_no)
  os.environ.pop(xenv.CPU_NUM_DEVICES, None)
  # XLA CPU has no support for cross-replica reduces, so we have to reduce using
  # torch.distributed capabilities. Since the logic is to use torch.distributed
  # across hosts (with XLA device reduces across devices within the same host),
  # we make the single host processes behave like if they were different hosts.
  os.environ[xenv.HOST_ORDINAL] = str(gindex)


def _wants_tpu_env_config(index, gindex):
  if xenv.HOST_ORDINAL in os.environ:
    return index == 0
  return gindex == 0


def _setup_tpu_worker(index, gindex, tpu_env_config):
  os.environ[xenv.MP_DEVICE] = 'TPU:{}'.format(
      _get_mp_device_ordinal(index, gindex))
  if xenv.LOCAL_WORKER not in os.environ:
    # The local worker can be missing for a 1 TPU host setup. Make sure we
    # always have one.
    assert tpu_env_config, '{} environment must be populated'.format(
        xenv.TPU_CONFIG)
    tpu_config = _parse_tpu_config(tpu_env_config)
    worker = list(tpu_config.values())[0]
    os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(worker.worker_name,
                                                   worker.ordinal)
  if not _wants_tpu_env_config(index, gindex):
    # In multi-processing mode, only the process handling the first device of
    # the master worker, will do TPU mesh initialization, so we need to remove
    # the environment configs which would prevent the client to be falling in
    # the mesh client config path.
    os.environ.pop(xenv.TPU_CONFIG, None)
    os.environ.pop(xenv.TPU_NUM_DEVICES, None)


def _prepare_env_for_index(index, pf_cfg):
  _setup_world_size(pf_cfg)
  gindex = _local_index_to_global(index, pf_cfg.num_devices)
  os.environ[xenv.ORDINAL] = str(gindex)
  os.environ[xenv.LOCAL_ORDINAL] = str(index)

  if pf_cfg.dev_kind == 'TPU':
    _setup_tpu_worker(index, gindex, os.environ.get(xenv.TPU_CONFIG, None))
    if xenv.HOST_ORDINAL in os.environ:
      # If xenv.HOST_ORDINAL is set, we are in a sea-of-devices TPU setup, where
      # each host has local TPU devices, but not interconnected with the fast TPU
      # link. In this case each TPU host sees only local TPU devices, as far as
      # fast TPU reduction goes, and global redcutions are performed with normal
      # torch.distributed facilities. The ordinal 0 of each TPU host will be the
      # one performing the global reductions, if no groups are defined in the
      # reduce operation.
      # Sea of devices configuration:
      #  - xenv.HOST_ORDINAL must be set to the host ordinal.
      #  - xenv.TORCH_DIST_ROOT must be set to the HOST:PORT, where HOST can be
      #    the same host of the mesh master, but different port.
      #  - xenv.TPU_CONFIG must be set on all host, with task number equal 0.
      #  - The worker ordinal (task number) in the xenv.LOCAL_WORKER must be set
      #    to 0 in all hosts.
      _setup_torch_distributed()
  elif pf_cfg.dev_kind == 'GPU':
    _setup_gpu_worker(index, gindex)
  elif pf_cfg.dev_kind == 'CPU':
    _setup_cpu_worker(index, gindex)
    _setup_torch_distributed()
  return gindex


def _setup_replication():
  # At this point xla_model.py APIs are allowed as the setup is already
  # completed.
  if xm.xrt_world_size() > 1:
    device = xm.xla_device()
    xm.set_replication(device, [device])


def _start_fn(index, pf_cfg, fn, args):
  gindex = _prepare_env_for_index(index, pf_cfg)
  # Calling _setup_replication() will trigger XLA library initialization, so the
  # environment must be fully setup before doing so.
  _setup_replication()
  fn(gindex, *args)


def _mp_start_fn(index, pf_cfg, fn, args):
  exit_code = 0
  try:
    _start_fn(index, pf_cfg, fn, args)
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
    start_method (string): The Python `multiprocessing` process creation method.
      Default: `spawn`

  Returns:
    The same object returned by the `torch.multiprocessing.spawn` API. If
    `nprocs` is 1 the `fn` function will be called directly, and the API will
    return None.
  """
  if pjrt.using_pjrt():
    return pjrt.spawn(fn, nprocs, start_method, args)

  if not _is_xla_config():
    # If this is not an XLA setup, jump to normal multi-processing.
    return _run_direct(fn, args, nprocs, join, daemon, start_method)

  pf_cfg = _pre_fork_setup(nprocs)
  result = None
  if pf_cfg.num_devices == 1:
    _start_fn(0, pf_cfg, fn, args)
  else:
    result = torch.multiprocessing.start_processes(
        _mp_start_fn,
        args=(pf_cfg, fn, args),
        nprocs=pf_cfg.num_devices,
        join=join,
        daemon=daemon,
        start_method=start_method)

  # For GPU, xenv.WORKERS are set in the launcher and then get carried to the children.
  # However, if the launcher is reused to do another multi-process experiment, _setup_gpu_workers
  # would mistake the xenv.WORKERS as configured to enable multi-host experiments. Each worker then
  # represents a host. Therefore, reset it after launching all children.
  if pf_cfg.dev_kind == 'GPU':
    os.environ.pop(xenv.WORKERS)

  return result


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

    xmp.spawn(_mp_fn, ..., start_method='fork')

  This method has two advantages. First it uses only one copy of the memory
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


class MpSerialExecutor(object):
  """Utility to run a function in a serialized fashion among multi-core processes.

  Example::

    # At global scope.
    SERIAL_EXEC = xmp.MpSerialExecutor()

    def load_dataset(path):
      return maybe_download_and_load(path)

    def _mp_fn(index, ...):
      # Avoid all cores downloading the same data with the serial executor.
      dataset = SERIAL_EXEC.run(lambda: load_dataset('/tmp/mnist-data'))
      ...

    xmp.spawn(_mp_fn, ...)
  """

  def __init__(self):
    self._lock = torch.multiprocessing.Lock()

  def run(self, fn):
    """Runs the provided function serialized WRT each per-core process.

    Args:
      fn (callable): The function to run in a serialized fashion.
    Returns:
      The `fn` return value.
    """
    with self._lock:
      return fn()
