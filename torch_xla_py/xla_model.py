from __future__ import print_function

import collections
import gc
from six import itervalues
import os
import re
import threading
import time
import torch
import torch.nn as nn
import torch_xla
import torch_xla_py.graph_saver as gs
import torch_xla_py.utils as xu
import torch_xla_py.keyd_queue as kq

_XLA_DEVICES = torch_xla._XLAC._xla_get_devices()
_XLA_ALL_DEVICES = torch_xla._XLAC._xla_get_all_devices()

_TLS = threading.local()


class Replication(object):

  def __init__(self, devices, replication_devices):
    self._devices = list(devices)
    self._replication_devices = list(replication_devices)
    self._lock = threading.Lock()
    self._ready_cv = threading.Condition(self._lock)
    self._join_count = 0
    self._ready = False

  def devices(self):
    return self._devices

  def replication_devices(self):
    return self._replication_devices

  def enter(self):
    with self._lock:
      self._join_count += 1
      if self._join_count == len(self._devices):
        self._ready = True
        self._ready_cv.notify_all()
      else:
        while not self._ready:
          self._ready_cv.wait()
      self._join_count -= 1
      if self._join_count == 0:
        self._ready = False


def set_replication(replication):
  assert replication is None or isinstance(replication, Replication)
  _TLS.replication = replication


def is_xla_tensor(tensor):
  return tensor.device.type == 'xla'


def parse_xla_device(device):
  m = re.match(r'(CPU|TPU|GPU):(\d+)$', device)
  if m:
    return (m.group(1), int(m.group(2)))


def get_xla_supported_devices(devkind=None, max_devices=None):
  devkind = devkind or ['TPU', 'GPU', 'CPU']
  for kind in devkind:
    kind_devices = []
    for i, device in enumerate(_XLA_DEVICES):
      if re.match(kind + r':\d+$', device):
        kind_devices.append('xla:{}'.format(i))
    if kind_devices:
      return kind_devices[:max_devices] if max_devices else kind_devices


def xla_device(n=None, devkind=None):
  if n is None:
    devices = get_xla_supported_devices(devkind=devkind)
    assert devices, 'No devices of {} kind'.format(devkind or 'ANY')
    return torch.device(devices[0])
  return torch.device('xla:{}'.format(n))


def xla_real_devices(devices):
  real_devices = []
  for device in devices:
    m = re.match(r'xla:(\d+)$', device)
    if m:
      real_devices.append(_XLA_DEVICES[int(m.group(1))])
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
    return None
  device_type = device_types.pop()
  kind_devices = get_xla_supported_devices(devkind=[device_type])
  if len(kind_devices) != len(local_devices):
    # Replication can only happen among all devices of one kind.
    return None
  replication_devices = []
  for device in _XLA_ALL_DEVICES:
    xdev = parse_xla_device(device)
    if not xdev:
      raise RuntimeError('Invalid device format: {}'.format(device))
    if xdev[0] == device_type:
      replication_devices.append(device)
  return replication_devices


class OptimizerState(object):

  def __init__(self):
    self.tensors = []
    self.gradients = []


class RateTracker(object):

  def __init__(self, smooth_factor=0.8):
    self._smooth_factor = smooth_factor
    self._start_time = time.time()
    self._partial_time = self._start_time
    self._count = 0
    self._rate = 0.0

  def update(self, count):
    now = time.time()
    delta = now - self._partial_time
    if delta > 0:
      rate = (count - self._count) / delta
      self._rate = (
          self._rate * self._smooth_factor + rate * (1.0 - self._smooth_factor))
    self._partial_time = now
    self._count = count
    return self._rate

  def add(self, count):
    return self.update(self._count + count)

  def rate(self):
    return self._rate

  def global_rate(self):
    return self._count / (self._partial_time - self._start_time)


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
    if self._select_fn(inputs):
      self._add(inputs)
    elif isinstance(inputs, (list, tuple, set)):
      for x in inputs:
        self._collect_tensors(x)
    elif isinstance(inputs, dict):
      for k, v in inputs.items():
        self._collect_tensors(k)
        self._collect_tensors(v)

  def _replace_tensors(self, inputs):
    if self._select_fn(inputs):
      return self._get_converted_tensor()
    elif isinstance(inputs, (list, tuple, set)):
      outputs = []
      for x in inputs:
        outputs.append(self._replace_tensors(x))
      return type(inputs)(outputs)
    elif isinstance(inputs, dict):
      outputs = {}
      for k, v in inputs.items():
        k = self._replace_tensors(k)
        v = self._replace_tensors(v)
        outputs[k] = v
      return outputs
    return inputs

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


def _fetch_optimizer_state(optimizer):

  def add(p, state):
    if isinstance(p, torch.Tensor):
      state.tensors.append(p.data)
      if p.grad is not None:
        state.gradients.append(p.grad.data)
    elif isinstance(p, dict):
      for k, v in p.items():
        add(k, state)
        add(v, state)
    elif isinstance(p, (list, tuple, set)):
      for x in p:
        add(x, state)

  state = OptimizerState()
  add(optimizer.__getstate__(), state)
  return state


def _mark_step(state, replication):
  save_dir = os.environ.get('SAVE_GRAPH_DIR', None)
  if save_dir:
    gs.save_tensors_graph(save_dir, 'optimizer_step',
                          state.gradients + state.tensors)
  devices = []
  if replication:
    replication.enter()
    devices = replication.replication_devices()
  torch_xla._XLAC._xla_step_marker(
      torch_xla._XLAC._xla_get_default_device(), devices, wait=False)


def optimizer_step(optimizer, closure=None):
  replication = getattr(_TLS, 'replication', None)
  state = _fetch_optimizer_state(optimizer)
  count = len(replication.replication_devices()) if replication else 1
  if count > 1:
    torch_xla._XLAC._xla_cross_replica_sum(state.gradients, 1.0 / count, [])
  loss = optimizer.step(closure=closure)
  _mark_step(state, replication)
  return loss
