from __future__ import division
from __future__ import print_function

import os
from six import iteritems, itervalues
from copy import deepcopy
import sys
import threading
import torch
import torch.autograd
import torch_xla
import torch_xla_py.keyd_queue as kq
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import traceback


class ThreadResult(object):

  def __init__(self):
    self.result = None


class PerDeviceQueue(object):

  def __init__(self, device, loader_prefetch_size, device_prefetch_size):
    self.device = device
    self.batch_number = 0
    self.loader_queue = kq.Queue(maxsize=loader_prefetch_size)
    self.queue = kq.Queue(maxsize=device_prefetch_size)


class PerDeviceLoader(object):

  def __init__(self, loader, device):
    self._loader = loader
    self._device = device

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    item = self._loader.next_item(self._device)
    if item is None:
      raise StopIteration
    return item


class ParallelLoader(object):

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               drop_last=False,
               loader_prefetch_size=8,
               device_prefetch_size=4):
    self._loader = loader
    self._batch_size = None
    self._devices = list(devices)
    self._batchdim = batchdim
    self._drop_last = drop_last
    self._done = False
    self._queues = dict()
    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for dqueue in itervalues(self._queues):
      thread = threading.Thread(target=self._worker, args=(dqueue,))
      thread.daemon = True
      thread.start()

  def per_device_loader(self, device):
    return PerDeviceLoader(self, device)

  def next_item(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in itervalues(self._queues):
      dqueue.queue.close()
      dqueue.loader_queue.close()

  def _get_batch_size(self, data, dim):
    size = []

    def fn(v):
      csize = v.size()[dim]
      if not size:
        size.append(csize)
      else:
        assert csize == size[0]

    xu.for_each_instance(data, torch.Tensor, fn)
    return size[0] if size else None

  def _send_data_to(self, data, device):

    def convert_fn(tensors):
      devices = [str(device)] * len(tensors)
      return torch_xla._XLAC._xla_tensors_from_aten(tensors, devices)

    def select_fn(v):
      return type(v) == torch.Tensor

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(data)

  def _loader_worker(self):
    num_batches = (len(self._loader) // len(self._devices)) * len(self._devices)
    batch_number = 0
    queues = list(self._queues.values())
    data_iter = enumerate(self._loader)
    while batch_number < num_batches and not self._done:
      try:
        _, data = next(data_iter)
      except StopIteration:
        break
      # There is only one loader worker thread, so it is safe to write the
      # batch_size in this way. Also, the batch_size is only referenced
      # within this thread.
      if self._batch_size is None:
        self._batch_size = self._get_batch_size(data, self._batchdim)
      elif (self._drop_last and
            self._batch_size != self._get_batch_size(data, self._batchdim)):
        break
      queues[batch_number % len(queues)].loader_queue.put((batch_number, data))
      batch_number += 1
    for dqueue in queues:
      dqueue.loader_queue.close_write()

  def _get_batch(self, dqueue):
    batch = []
    while dqueue.queue.max_size() > len(batch):
      item = dqueue.loader_queue.get()
      if item is None:
        break
      batch.append(item[1])
    return batch

  def _worker(self, dqueue):
    device = torch.device(dqueue.device)
    while True:
      batch = self._get_batch(dqueue)
      if not batch:
        break
      batch = self._send_data_to(batch, device)
      for data in batch:
        dqueue.queue.put((dqueue.batch_number, data))
        dqueue.batch_number += 1
    dqueue.queue.close_write()


class DataParallel(object):

  def __init__(self, network, device_ids=None, batchdim=0, drop_last=False):
    if device_ids is None:
      device_ids = xm.get_xla_supported_devices()
    self._device_ids = list(device_ids)
    self._batchdim = batchdim
    self._drop_last = drop_last
    replication_devices = (
        xm.xla_replication_devices(self._device_ids)
        if self._device_ids else None)
    self._replication = (
        xm.Replication(self._device_ids, replication_devices)
        if replication_devices else None)
    self._models = []
    module = network if isinstance(network, torch.nn.Module) else network()
    for device in device_ids:
      device_module = deepcopy(module).to(device=torch.device(device))
      self._models.append(device_module)
    if not self._models:
      # No XLA device, push a vanilla network in.
      self._models.append(network)
      self._device_ids.append(self._get_model_device(network))

  def _get_model_device(self, model):
    devices = {str(p.device) for p in model.parameters()}
    if len(devices) > 1:
      raise RuntimeError('Model uses more than one device: {}'.format(
          list(devices)))
    return devices.pop() if devices else 'cpu'

  def _handle_runner_exception(self, device, e):
    print(
        'Exception in model function for device={}: {}'.format(device, str(e)),
        file=sys.stderr)
    traceback.print_exc(limit=16, file=sys.stderr)
    # One exception in one thread is fatal, as the other ones (assuming they
    # somehow did not generate the same exception) will be getting stuck in
    # cross replica sum operations waiting for the defunct thread and its
    # device.
    os._exit(17)

  def _module_runner(self, loop_fn, device, module, loader, context, result):
    torch_xla._XLAC._xla_set_default_device(device)
    xm.set_replication(self._replication)
    try:
      result.result = loop_fn(module, loader, torch.device(device), context)
    except Exception as e:
      result.result = e
      self._handle_runner_exception(device, e)

  def __call__(self, loop_fn, loader):
    context = dict()
    if self._replication is None:
      ## This is called without XLA devices available. Run in normal mode.
      return [
          loop_fn(self._models[0], enumerate(loader),
                  torch.device(self._device_ids[0]), context)
      ]

    para_loader = ParallelLoader(
        loader,
        self._device_ids,
        batchdim=self._batchdim,
        drop_last=self._drop_last)
    threads = []
    results = []
    for module, device in zip(self._models, self._device_ids):
      result = ThreadResult()
      loader = para_loader.per_device_loader(device)
      thread = threading.Thread(
          target=self._module_runner,
          args=(loop_fn, device, module, loader, context, result))
      thread.daemon = True
      thread.start()
      threads.append(thread)
      results.append(result)
    for thread in threads:
      thread.join()
    para_loader.close()
    return [x.result for x in results]
