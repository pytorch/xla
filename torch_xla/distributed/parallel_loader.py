from __future__ import division
from __future__ import print_function

import multiprocessing.dummy
import threading
import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.utils.keyd_queue as kq


class ParallelLoader(object):

  def __init__(self, loader, batch_size, devices, prefetch_size=4, batchdim=0):
    self._loader = loader
    self._prefetch_size = prefetch_size
    self._batch_size = batch_size
    self._devices = list(devices)
    self._device_indices = dict()
    for i, device in enumerate(self._devices):
      self._device_indices[device] = i
    self._batch_number = 0
    self._batchdim = batchdim
    self._done = False
    self._lock = threading.Lock()
    self._loader_queue = kq.Queue(maxsize=self._prefetch_size)
    self._queue = kq.KeydQueue(maxsize=self._prefetch_size)
    self._worker_count = 0
    self._data = None
    self._device_slices = None
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for _ in range(0, prefetch_size):
      thread = threading.Thread(target=self._worker)
      thread.daemon = True
      thread.start()

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    item = self._queue.get(self._batch_number)
    if item is None:
      raise StopIteration
    self._data, target, self._device_slices = item
    self._batch_number += 1
    return self._batch_number - 1, (self._data, target)

  def close(self):
    self._done = True
    self._queue.close()
    self._loader_queue.close()

  def to(self, data, device):
    assert data is self._data
    return self._device_slices[self._device_indices[device]]

  def _up_workers(self, count):
    with self._lock:
      self._worker_count += count
      return self._worker_count

  def _loader_worker(self):
    batch_number = 0
    for (data, target) in self._loader:
      if data.size()[self._batchdim] != self._batch_size or self._done:
        break
      self._loader_queue.put((batch_number, (data, target)))
      batch_number += 1
    self._loader_queue.close_write()

  def _create_tensor_slices(self, data):
    if isinstance(data, torch.Tensor):
      mini_batch_size = self._batch_size // len(self._devices)
      slices = []
      for x in range(0, self._batch_size, mini_batch_size):
        slices.append(data[x:x + mini_batch_size])
      return slices
    elif isinstance(data, (list, tuple)):
      slices = []
      for xdata in data:
        slices.append(self._create_tensor_slices(xdata))
      return list(zip(*slices))
    else:
      raise RuntimeError('Unsupported input type: {}'.format(type(data)))

  def _send_to_devices(self, slices, pool):

    def _send(i):
      return slices[i].to(device=torch.device(self._devices[i]))

    if isinstance(slices[0], torch.Tensor):
      return pool.map(_send, range(0, len(slices)))
    elif isinstance(slices[0], (list, tuple)):
      device_slices = []
      for xslice in slices:
        device_slices.append(self._send_to_devices(xslice, pool))
      return list(zip(*device_slices))
    else:
      raise RuntimeError('Unsupported input type: {}'.format(type(slices[0])))

  def _worker(self):
    pool = multiprocessing.dummy.Pool(len(self._devices))
    self._up_workers(1)
    while True:
      item = self._loader_queue.get()
      if item is None:
        break
      batch_number, (data, target) = item
      slices = self._create_tensor_slices(data)
      device_slices = self._send_to_devices(slices, pool)
      self._queue.put(batch_number, (data, target, device_slices))
    if self._up_workers(-1) == 0:
      self._queue.close_write()
