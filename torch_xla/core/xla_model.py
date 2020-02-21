from __future__ import print_function

import collections
import sys
import os
import re
import threading
import time
import torch
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.debug.metrics_saver as ms
import torch_xla.utils.utils as xu
import torch_xla.utils.keyd_queue as kq

_TLS = threading.local()


def is_xla_tensor(tensor):
  return tensor.device.type == 'xla'


def parse_xla_device(device):
  m = re.match(r'(CPU|TPU|GPU):(\d+)$', device)
  if m:
    return (m.group(1), int(m.group(2)))


def get_xla_supported_devices(devkind=None, max_devices=None):
  """Returns a list of supported devices of a given kind.

  Args:
    devkind (string..., optional): If specified, one of `TPU`, `GPU` or `CPU`
      (the 'GPU' XLA device is currently not implemented).
    max_devices (int, optional): The maximum number of devices to be returned of
      that kind.

  Returns:
    The list of device strings.
  """

  xla_devices = torch_xla._XLAC._xla_get_devices()
  devkind = devkind or ['TPU', 'GPU', 'CPU']
  for kind in devkind:
    kind_devices = []
    for i, device in enumerate(xla_devices):
      if re.match(kind + r':\d+$', device):
        kind_devices.append('xla:{}'.format(i))
    if kind_devices:
      return kind_devices[:max_devices] if max_devices else kind_devices


def xrt_world_size(defval=1):
  """Retrieves the number of devices which is taking part of the replication.

  Args:
    defval (int, optional): The default value to be returned in case there is no
      replication information available.
      Default: 1

  Returns:
    The number of devices which is taking part of the replication.
  """

  return xu.getenv_as(xenv.WORLD_SIZE, int, defval=defval)


def get_ordinal(defval=0):
  """Retrieves the replication ordinal of the current process.

  The ordinals range from 0 to `xrt_world_size()` minus 1.

  Args:
    defval (int, optional): The default value to be returned in case there is no
      replication information available.
      Default: 0

  Returns:
    The replication ordinal of the current process.
  """

  return xu.getenv_as(xenv.ORDINAL, int, defval=defval)


def get_local_ordinal(defval=0):
  """Retrieves the replication local ordinal of the current process.

  The local ordinals range from 0 to the number of local devices minus 1.

  Args:
    defval (int, optional): The default value to be returned in case there is no
      replication information available.
      Default: 0

  Returns:
    The replication local ordinal of the current process.
  """

  ordinal = xu.getenv_as(xenv.LOCAL_ORDINAL, int, defval=-1)
  if ordinal >= 0:
    return ordinal
  return getattr(_TLS, 'device_index', defval)


def is_master_ordinal(local=True):
  """Checks whether the current process is the master ordinal (0).

  Args:
    local (bool): Whether the local or global master ordinal should be checked.
      In case of multi-host replication, there is only one global master ordinal
      (host 0, device 0), while there are NUM_HOSTS local master ordinals.
      Default: True

  Returns:
    A boolean indicating whether the current process is the master ordinal.
  """

  ordinal = get_local_ordinal() if local else get_ordinal()
  return ordinal == 0


def master_print(s, fd=sys.stdout, local=True):
  if is_master_ordinal(local=local):
    print(s, file=fd)


def xla_device(n=None, devkind=None):
  """Returns a given instance of an XLA device.

  Args:
    n (int, optional): The specific instance (ordinal) to be returned. If
      specified, the specific XLA device instance will be returned. Otherwise
      the first device of `devkind` will be returned.
    devkind (string..., optional): If specified, one of `TPU`, `GPU` or `CPU`
      (the 'GPU' XLA device is currently not implemented).

  Returns:
    A `torch.device` with the requested instance.
  """

  if n is None:
    devices = get_xla_supported_devices(devkind=devkind)
    assert devices, 'No devices of {} kind'.format(devkind or 'ANY')
    # This is a utility API mainly called from tests or simple code which wants
    # to just have a single device to run on. Set the default device so that
    # the tensor barrier can work correctly and avoid growing graphs surprises.
    device = devices[0]
  else:
    device = 'xla:{}'.format(n)
  torch_xla._XLAC._xla_set_default_device(device)
  return torch.device(device)


def xla_real_devices(devices):
  xla_devices = torch_xla._XLAC._xla_get_devices()
  real_devices = []
  for device in devices:
    m = re.match(r'xla:(\d+)$', device)
    if m:
      real_devices.append(xla_devices[int(m.group(1))])
      continue
    xdev = parse_xla_device(device)
    if not xdev:
      raise RuntimeError('Invalid device format: {}'.format(device))
    real_devices.append(device)
  return real_devices


def xla_replication_devices(local_devices):
  real_devices = xla_real_devices(local_devices)
  device_types = set()
  for device in real_devices:
    xdev = parse_xla_device(device)
    device_types.add(xdev[0])
  if len(device_types) != 1:
    # No replication if the device set spawns multiple device types.
    raise RuntimeError(
        'Cannot replicate across different device types: devices={}/{}'.format(
            local_devices, real_devices))
  device_type = device_types.pop()
  kind_devices = get_xla_supported_devices(devkind=[device_type])
  if len(kind_devices) != len(local_devices):
    # Replication can only happen among all devices of one kind.
    raise RuntimeError(
        'Cannot replicate if number of devices ({}) is different from {}'
        .format(len(local_devices), len(kind_devices)))
  replication_devices = []
  for device in torch_xla._XLAC._xla_get_all_devices():
    xdev = parse_xla_device(device)
    if not xdev:
      raise RuntimeError('Invalid device format: {}'.format(device))
    if xdev[0] == device_type:
      replication_devices.append(device)
  return replication_devices


def set_replication(device, devices):
  if devices:
    replication_devices = xla_replication_devices(devices)
    torch_xla._XLAC._xla_set_replication_devices(replication_devices)
    _TLS.device_index = devices.index(device)
  else:
    torch_xla._XLAC._xla_set_replication_devices([])
    _TLS.device_index = 0
  _TLS.device = device
  _TLS.all_reduce_token = None
  torch_xla._XLAC._xla_set_default_device(device)


class RateTracker(object):

  def __init__(self, smooth_factor=None):
    self._smooth_factor = xu.getenv_as(
        'RATE_TRACKER_SMOOTHING', float,
        0.4) if smooth_factor is None else smooth_factor
    self._start_time = time.time()
    self._partial_time = self._start_time
    self._partial_count = 0.0
    self._partial_rate = None
    self._count = 0.0

  def _update(self, now, rate):
    self._partial_count += self._count
    self._count = 0.0
    self._partial_time = now
    self._partial_rate = rate

  def add(self, count):
    self._count += count

  def _smooth(self, current_rate):
    if self._partial_rate is None:
      smoothed_rate = current_rate
    else:
      smoothed_rate = ((1 - self._smooth_factor) * current_rate +
                       self._smooth_factor * self._partial_rate)
    return smoothed_rate

  def rate(self):
    now = time.time()
    delta = now - self._partial_time
    report_rate = 0.0
    if delta > 0:
      report_rate = self._smooth(self._count / delta)
      self._update(now, report_rate)
    return report_rate

  def global_rate(self):
    delta = time.time() - self._start_time
    count = self._partial_count + self._count
    return count / delta if delta > 0 else 0.0


class ToXlaTensorArena(object):

  def __init__(self, convert_fn, select_fn):
    self._convert_fn = convert_fn
    self._select_fn = select_fn
    self._tensors = []

  def _add(self, tensor):
    self._tensors.append(tensor)

  def _convert(self):
    self._index = 0
    if self._tensors:
      self._converted_tensors = self._convert_fn(self._tensors)
    else:
      self._converted_tensors = []

  def _get_converted_tensor(self):
    assert self._index < len(self._converted_tensors)
    new_tensor = self._converted_tensors[self._index]
    self._index += 1
    return new_tensor

  def _collect_tensors(self, inputs):

    def collect_fn(value):
      self._add(value)

    xu.for_each_instance(inputs, lambda x: self._select_fn(x), collect_fn)

  def _replace_tensors(self, inputs):

    def convert_fn(value):
      return self._get_converted_tensor()

    return xu.for_each_instance_rewrite(inputs, lambda x: self._select_fn(x),
                                        convert_fn)

  def transform(self, inputs):
    self._tensors = []
    self._collect_tensors(inputs)
    self._convert()
    return self._replace_tensors(inputs)


def check_view_sharing(obj):
  tensors = set()
  aliases = dict()

  def tensor_info(t):
    return '{}{}'.format(t.dtype, list(t.size()))

  def tensor_id(t):
    if is_xla_tensor(t):
      return torch_xla._XLAC._xla_get_tensor_id(t), 'xla'
    return id(t), 'torch'

  def alias_id(t):
    if is_xla_tensor(t):
      aid = torch_xla._XLAC._xla_get_tensor_view_alias_id(t)
      return None if aid == 0 else aid, 'xla'
    return t.storage().data_ptr(), 'torch'

  def check_object(obj):
    tid = tensor_id(obj)
    if tid not in tensors:
      tensors.add(tid)
      aid = alias_id(obj)
      if aid[0] is not None:
        if aid in aliases:
          oobj = aliases[aid]
          raise RuntimeError(
              'Tensor ID {} ({}) is sharing a view with tensor ID {} ({})'
              .format(tid, tensor_info(obj), tensor_id(oobj),
                      tensor_info(oobj)))
        aliases[aid] = obj

  xu.for_each_instance(obj, lambda x: type(x) == torch.Tensor, check_object)


def _fetch_gradients(optimizer):
  gradients = []
  for param_group in optimizer.__getstate__()['param_groups']:
    for group, params in param_group.items():
      if group == 'params':
        for p in params:
          if isinstance(p, torch.Tensor) and p.grad is not None:
            gradients.append(p.grad.data)
  return gradients


def _get_all_reduce_token():
  token = getattr(_TLS, 'all_reduce_token', None)
  if token is None:
    token = torch_xla._XLAC._xla_create_token()
    _TLS.all_reduce_token = token
  return token


def all_reduce(reduce_type, inputs, scale=1.0, groups=[]):
  """Perform an inplace reduce operation on the input tensors.

  Args:
    reduce_type (string): One of ``sum``, ``mul``, ``and``, ``or``, ``min`` and
      ``max``.
    inputs (list): List of tensors to perform the all reduce op to.
    scale (float): A default scaling value to be applied after the reduce.
      Default: 1.0
    groups (list): Reserved.
  """
  _TLS.all_reduce_token = torch_xla._XLAC._xla_all_reduce(
      reduce_type, inputs, _get_all_reduce_token(), scale, groups)


def add_step_closure(closure, args=()):
  """Adds a closure to the list of the ones to be run at the end of the step.

  Many times during model training there is the need to print/report (print to
  console, post to tensorboard, etc...) information which require the content of
  intermediary tensors to be inspected.
  Inspecting different tensors content in different points of the model code
  requires many executions and typically causes performance issues.
  Adding a step closure will ensure that it will be run after the barrier, when
  all the live tensors will be already materialized to device data.
  Live tensors which will include the ones captured by the closure arguments.
  So using `add_step_closure()` will ensure a single execution will be
  performed, even when multiple closures are queued, requiring multiple tensors
  to be inspected.
  Step closures will be run sequentially in the order they have been queued.
  Note that even though using this API the execution will be optimized, it is
  advised to throttle the printing/reporting events once every N steps.

  Args:
    closure (callable): The function to be called.
    args (tuple): The arguments to be passed to the closure.
  """
  step_closures = getattr(_TLS, 'step_closures', None)
  if step_closures is None:
    step_closures = []
    _TLS.step_closures = step_closures
  step_closures.append(lambda a=args: closure(*a))


def _run_step_closures():
  step_closures = getattr(_TLS, 'step_closures', None)
  if step_closures is not None:
    _TLS.step_closures = []
    for closure in step_closures:
      closure()


def mark_step():
  if xu.getenv_as('XLA_EMIT_STEPLOG', bool, False):
    print('torch_xla.core.xla_model::mark_step', file=sys.stderr, flush=True)
  torch_xla._XLAC._xla_step_marker(
      torch_xla._XLAC._xla_get_default_device(), [],
      wait=xu.getenv_as('XLA_SYNC_WAIT', bool, False))
  # Only emit metrics from the first local device index, to avoid emitting the
  # same values from different threads.
  if is_master_ordinal():
    ms.save_metrics()
  _run_step_closures()
  _TLS.all_reduce_token = None


def wait_device_ops(devices=[]):
  """Waits for all the async operations on the given devices to complete.

  Args:
    devices (string..., optional): The devices whose async ops need to be waited
      for. If empty, all the local devices will be waited for.
  """
  torch_xla._XLAC._xla_wait_device_ops(devices=devices)


def optimizer_step(optimizer, barrier=False, optimizer_args={}):
  """Run the provided optimizer step and issue the XLA device step computation.

  Args:
    optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance whose
      `step()` function needs to be called. The `step()` function will be called
      with the `optimizer_args` named arguments.
    barrier (bool, optional): Whether the XLA tensor barrier should be issued in
      this API. If using the PyTorch XLA `ParallelLoader` or `DataParallel`
      support, this is not necessary as the barrier will be issued by the XLA
      data loader iterator `next()` call.
      Default: False
    optimizer_args (dict, optional): Named arguments dictionary for the
      `optimizer.step()` call.

  Returns:
    The same value returned by the `optimizer.step()` call.
  """

  count = torch_xla._XLAC._xla_get_replication_devices_count()
  if count > 1:
    gradients = _fetch_gradients(optimizer)
    all_reduce('sum', gradients, scale=1.0 / count)
  loss = optimizer.step(**optimizer_args)
  if barrier:
    mark_step()
  return loss


def save(data, file_or_path, master_only=True):
  """Saves the input data into a file.

  The saved data is transfered to PyTorch CPU device before being saved, so a
  following `torch.load()` will load CPU data.

  Args:
    data: The input data to be saved. Any nested combination of Python objects
      (list, tuples, sets, dicts, ...).
    file_or_path: The destination for the data saving operation. Either a file
      path or a Python file object. If `master_only` is ``False`` the path or
      file objects must point to different destinations as otherwise all the
      writes from the same host will override each other.
    master_only (bool): Whether only the master device should save the data. If
      False, the `file_or_path` argument should be a different file or path for
      each of the ordinals taking part to the replication, otherwise all the
      replicas on the same host will be writing to the same location.
      Default: True
  """

  def convert_fn(tensors):
    cpu_tensors = []
    torch_xla._XLAC._xla_sync_multi(
        tensors, devices=[], wait=True, sync_xla_data=True)
    for sync_tensor in tensors:
      cpu_tensors.append(sync_tensor.cpu())
    return cpu_tensors

  def select_fn(v):
    return type(v) == torch.Tensor and is_xla_tensor(v)

  cpu_data = ToXlaTensorArena(convert_fn, select_fn).transform(data)
  if master_only:
    if is_master_ordinal():
      torch.save(cpu_data, file_or_path)
  else:
    torch.save(cpu_data, file_or_path)
  rendezvous('torch_xla.core.xla_model.save')


def send_cpu_data_to_device(data, device):

  def convert_fn(tensors):
    devices = [str(device)] * len(tensors)
    return torch_xla._XLAC._xla_tensors_from_aten(tensors, devices)

  def select_fn(v):
    return type(v) == torch.Tensor and v.device.type == 'cpu'

  return ToXlaTensorArena(convert_fn, select_fn).transform(data)


def rendezvous(tag, payload=''):
  """Waits for all the mesh clients to reach the named rendezvous.

  Args:
    tag (string): The name of the rendezvous to join.
    payload (string, optional): The payload to be sent to the rendezvous.
  """
  return torch_xla._XLAC._xla_rendezvous(get_ordinal(), tag, payload)
