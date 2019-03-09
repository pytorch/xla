from __future__ import division
from __future__ import print_function

from six import iteritems, itervalues
import threading
import torch
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torch_xla_py.utils as xu
import torch_xla_py.keyd_queue as kq


class ThreadResult(object):

  def __init__(self):
    self.result = None


class PerDeviceQueue(object):

  def __init__(self, device, maxsize):
    self.device = device
    self.batch_number = 0
    self.queue = kq.Queue(maxsize=maxsize)


class PerDeviceLoader(object):

  def __init__(self, loader, device):
    self._loader = loader
    self._device = device

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    item = self._loader.next(self._device)
    if item is None:
      raise StopIteration
    return item


class ParallelLoader(object):

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               loader_prefetch_size=8,
               device_prefetch_size=4):
    self._loader = loader
    self._batch_size = None
    self._devices = list(devices)
    self._batchdim = batchdim
    self._done = False
    self._lock = threading.Lock()
    self._loader_queue = kq.Queue(maxsize=loader_prefetch_size)
    self._queues = dict()
    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, device_prefetch_size)
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for dqueue in itervalues(self._queues):
      thread = threading.Thread(target=self._worker, args=(dqueue,))
      thread.daemon = True
      thread.start()

  def per_device_loader(self, device):
    return PerDeviceLoader(self, device)

  def next(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in itervalues(self._queues):
      dqueue.queue.close()
    self._loader_queue.close()

  def _expand_sample_batch(self, data, target):
    # TODO: Expand last sample,target to batch size
    return data, target

  def _loader_worker(self):
    # TODO: When _expand_sample_batch() is implemented, remove the -1 fixup.
    loader_batches = max(len(self._loader) - 1, 0)
    num_batches = (loader_batches // len(self._devices)) * len(self._devices)
    batch_number = 0
    while batch_number < num_batches and not self._done:
      try:
        data, target = self._loader.next()
      except StopIteration:
        break
      # There is only one loader worker thread, so it is safe to write the
      # batch_size in this way. Also, the batch_size is only referenced
      # within this thread.
      if self._batch_size is None:
        self._batch_size = data.size()[self._batchdim]
      if data.size()[self._batchdim] != self._batch_size:
        data, target = self._expand_sample_batch(data, target)
      self._loader_queue.put((batch_number, (data, target)))
      batch_number += 1
    self._loader_queue.close_write()

  def _worker(self, dqueue):
    while True:
      item = self._loader_queue.get()
      if item is None:
        break
      batch_number, (data, target) = item
      data = data.to(device=torch.device(dqueue.device))
      target = target.to(device=torch.device(dqueue.device))
      dqueue.queue.put((dqueue.batch_number, (data, target)))
      dqueue.batch_number += 1
    dqueue.queue.close_write()


class DataParallel(object):

  def __init__(self, network, loader, loop_fn, device_ids=None, batchdim=0):
    if not device_ids:
      device_ids = xm.get_xla_supported_devices()
    self._loader = loader
    self._loop_fn = loop_fn
    self._batchdim = batchdim
    self._device_ids = list(device_ids)
    self._para_loader = None
    if self._device_ids:
      self._para_loader = ParallelLoader(
          self._loader, self._device_ids, batchdim=self._batchdim)
    self._modules = []
    for device in device_ids:
      module = network().to(device=torch.device(device))
      self._modules.append(module)
    if not self._modules:
      # No XLA device, push a vanilla network in.
      self._modules.append(network())

  def _module_runner(self, device, module, loader, result):
    torch_xla._XLAC._xla_set_default_device(device)
    result.result = self._loop_fn(module, loader)

  def __call__(self):
    if not self._device_ids:
      ## This is called without XLA devices available. Run in normal mode.
      return self._loop_fn(self._modules[0], self._loader)
    threads = []
    results = []
    for module, device in zip(self._modules, self._device_ids):
      result = ThreadResult()
      loader = self._para_loader.per_device_loader(device)
      thread = threading.Thread(
          target=self._module_runner, args=(device, module, loader, result))
      thread.daemon = True
      thread.start()
      threads.append(thread)
      results.append(result)
    for thread in threads:
      thread.join()
    return [x.result for x in results]
