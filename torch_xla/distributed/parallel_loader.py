from __future__ import division
from __future__ import print_function

from six import iteritems, itervalues
import threading
import torch
import torch_xla
import torch_xla.utils.keyd_queue as kq
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


def _get_input_size(data):
  obj = xu.Object({'size': 0})

  def fn(v):
    obj.size += v.numel() * v.element_size()

  xu.for_each_instance(data, lambda x: type(x) == torch.Tensor, fn)
  return obj.size


class PerDeviceQueue(object):

  def __init__(self, device, loader_prefetch_size, device_prefetch_size):
    self.device = device
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
    xm.mark_step()
    item = self._loader.next_item(self._device)
    if item is None:
      raise StopIteration
    return item


class ParallelLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    devices (`torch.device`...): The list of devices where the data has to be
      sent. The i-th sample returned by the `loader` will be sent to `devices[i
      % len(devices)]`.
    batchdim (int, optional): The dimension which is holding the batch size.
      Default: 0
    fixed_batch_size (bool, optional): Ensures that all the batch sizes sent to
      the devices are of the same size. The original `loader` iteration stops as
      soon as a not matching batch size is found.
      Default: False
    loader_prefetch_size (int, optional): The max capacity of the queue used by
      the thread which is reading samples from the `loader`, to be processed by
      the worker threads which upload data to the devices.
      Default: 8
    device_prefetch_size (int, optional): The max size of the per-device queues,
      where the worker threads deposit tensors which have already been sent to
      devices.
      Default: 4
    max_device_prefetch_mem (int, optional): The maximum memory a single device
      data upload can use.
      Default: 1500000000
  """

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               fixed_batch_size=False,
               loader_prefetch_size=8,
               device_prefetch_size=4,
               max_device_prefetch_mem=1500000000):
    self._loader = loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._fixed_batch_size = fixed_batch_size
    self._max_device_prefetch_mem = max_device_prefetch_mem
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
    """Retrieves the loader iterator object for the given device.

    Args:
      device (`torch.device`): The device whole loader is being requested.

    Returns:
      The loader iterator object for the `device`. This is not a
      `torch.utils.data.DataLoader` interface, but a Python iterator which
      returns the same tensor data structure as returned by the wrapped
      `torch.utils.data.DataLoader`, but residing on XLA devices.
    """
    return PerDeviceLoader(self, torch.device(device))

  def next_item(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in itervalues(self._queues):
      dqueue.queue.close()
      dqueue.loader_queue.close()

  def _get_batch_size(self, data, dim):
    obj = xu.Object({'size': -1})

    def fn(v):
      csize = v.size()[dim]
      if obj.size < 0:
        obj.size = csize
      else:
        assert csize == obj.size

    xu.for_each_instance(data, lambda x: type(x) == torch.Tensor, fn)
    return obj.size if obj.size >= 0 else None

  def _loader_worker(self):
    queues = list(self._queues.values())
    data_iter = enumerate(self._loader)
    batch_size = None
    batch = []
    while not self._done:
      try:
        _, data = next(data_iter)
      except StopIteration:
        break
      if self._fixed_batch_size:
        if batch_size is None:
          batch_size = self._get_batch_size(data, self._batchdim)
        elif batch_size != self._get_batch_size(data, self._batchdim):
          break
      batch.append(data)
      if len(batch) == len(self._devices):
        for queue_no, device_batch in enumerate(batch):
          queues[queue_no].loader_queue.put(device_batch)
        batch = []
    for dqueue in queues:
      dqueue.loader_queue.close_write()

  def _get_batch(self, dqueue):
    item_size, mem_size = 0, 0
    batch = []
    while dqueue.queue.max_size() > len(batch):
      if batch and (mem_size + item_size) > self._max_device_prefetch_mem:
        break
      item = dqueue.loader_queue.get()
      if item is None:
        break
      if item_size == 0:
        item_size = _get_input_size(item)
      mem_size += item_size
      batch.append(item)
    return batch

  def _worker(self, dqueue):
    device = torch.device(dqueue.device)
    while True:
      batch = self._get_batch(dqueue)
      if not batch:
        break
      batch = xm.send_cpu_data_to_device(batch, device)
      for data in batch:
        dqueue.queue.put(data)
    dqueue.queue.close_write()
