from __future__ import print_function

import collections
import gc
from six import itervalues
import sys
import os
import re
import threading
import time
import torch
import torch.nn as nn
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


def is_master_ordinal():
  """Checks whether the current process is the master ordinal (0).

  Returns:
    A boolean indicating whether the current process is the master ordinal.
  """

  ordinal = get_ordinal(defval=-1)
  if ordinal >= 0:
    # We are either on multi-processing, or on BigSlice (or both).
    return ordinal == 0
  # We are in the multi-threaded DataParallel setup.
  return getattr(_TLS, 'device_index', 0) == 0


def master_print(s, fd=sys.stdout):
  if is_master_ordinal():
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


class TrainStepMetrics(object):

  LOG_FORMAT = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                'Loss: {:.6f}\tSamples/sec: {:.1f}')

  def __init__(self, epoch, num_cores, batch_number, num_batches, batch_size,
               loss, examples_per_sec, global_step):
    """Constructor for the metrics of a single train step.

    Args:
      epoch: The current epoch number.
      num_cores: The number of cores on which model is being trained.
      batch_number: The current batch number. Reset to 0 every epoch.
      num_batches: The number of batches in a single epoch.
      batch_size: Per core batch size.
      loss: Training loss.
      examples_per_sec: The number of processed samples per second.
      global_step: The global step number of current batch.
    """
    self._epoch = epoch
    self._processed_samples = num_cores * (batch_number + 1) * batch_size
    self._dataset_size = num_batches * batch_size
    self._percent_epoch_done = 100. * batch_number * num_cores / num_batches
    self._loss = loss
    self._examples_per_sec = examples_per_sec
    self._global_step = global_step
    self._global_step_per_sec = examples_per_sec / batch_size

  def write_summary(self, writer):
    if writer:
      writer.add_scalar('loss', self._loss, self._global_step)
      writer.add_scalar('global_step/sec', self._global_step_per_sec,
                        self._global_step)

  def __repr__(self):
    return self.LOG_FORMAT.format(self._epoch, self._processed_samples,
                                  self._dataset_size, self._percent_epoch_done,
                                  self._loss, self._examples_per_sec)


class TestStepMetrics(object):

  LOG_FORMAT = ('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), '
                'Samples/sec: {:.1f}\n')

  def __init__(self, loss, correct, count, examples_per_sec, global_step):
    """Constructor for the metrics of a single test step.

    Args:
      loss: The test loss.
      correct: The number of correct samples.
      count: Total number of samples.
      examples_per_sec: The number of processed samples per second.
      global_step: The global step number of current batch.
    """
    self._loss = loss
    self._correct = correct
    self._total = count
    self._global_step = global_step
    self._accuracy = 100.0 * correct / count
    self._examples_per_sec = examples_per_sec

  def write_summary(self, writer):
    if writer:
      writer.add_scalar('accuracy', self._accuracy, self._global_step)

  def __repr__(self):
    return self.LOG_FORMAT.format(self._loss, self._correct, self._total,
                                  self._accuracy, self._examples_per_sec)


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


def _get_summary_writer(logdir=None):
  if logdir:
    from tensorboardX import SummaryWriter
    return SummaryWriter(logdir)


def get_log_fn(logdir=None, custom_log_fn=print):
  writer = _get_summary_writer(logdir)

  def log_fn(step_result):
    if (isinstance(step_result, TrainStepMetrics) or
        isinstance(step_result, TestStepMetrics)):
      step_result.write_summary(writer)
      custom_log_fn(str(step_result))
    else:
      custom_log_fn(step_result)

  return log_fn


def check_view_sharing(obj):
  tensors = set()
  aliases = dict()

  def check_object(obj):
    if is_xla_tensor(obj):
      tid = torch_xla._XLAC._xla_get_tensor_id(obj)
      if tid not in tensors:
        tensors.add(tid)
        aid = torch_xla._XLAC._xla_get_tensor_view_alias_id(obj)
        if aid != 0:
          if aid in aliases:
            oobj = aliases[aid]
            raise RuntimeError(
                'Tensor ID {} is sharing a view with tensor ID {}'.format(
                    tid, torch_xla._XLAC._xla_get_tensor_id(oobj)))
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


def mark_step():
  torch_xla._XLAC._xla_step_marker(
      torch_xla._XLAC._xla_get_default_device(), [],
      wait=xu.getenv_as('XLA_SYNC_WAIT', bool, False))
  # Only emit metrics from the first local device index, to avoid emitting the
  # same values from different threads.
  if is_master_ordinal():
    ms.save_metrics()


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
    torch_xla._XLAC._xla_cross_replica_sum(gradients, 1.0 / count, [])
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
      path or a Python file object.
    master_only (bool): Whether only the master device should save the data. If
      False, the `file_or_path` argument must be a path, and the different
      devices will save data to files with their ordinal as extension.
      Default: True
  """

  def convert_fn(value):
    return value.cpu()

  cpu_data = xu.for_each_instance_rewrite(data,
                                          lambda x: type(x) == torch.Tensor,
                                          convert_fn)
  if master_only:
    if is_master_ordinal():
      torch.save(cpu_data, file_or_path)
  else:
    assert type(file_or_path) == str
    file_or_path = '{}.{}'.format(file_or_path, get_ordinal())
    torch.save(cpu_data, file_or_path)
