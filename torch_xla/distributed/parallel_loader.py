import itertools
import queue
import threading
import torch
import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.utils.keyd_queue as kq
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp


class PerDeviceQueue(object):

  def __init__(self, device, loader_prefetch_size, device_prefetch_size):
    self.device = device
    self.cpu_loader_queue = kq.Queue(maxsize=loader_prefetch_size)
    self.queue = kq.Queue(maxsize=device_prefetch_size)
    self.close_queue_count = itertools.count()


class PerDeviceLoader(object):

  def __init__(self, loader, device):
    self._loader = loader
    self._device = device
    self._mark_step_batch_count = loader.batches_per_execution - 1
    self._batches_yielded = 0

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def __len__(self):
    return self._loader.per_device_samples()

  def next(self):
    if xp.get_tracer_marked_step():
      xp.set_tracer_marked_step(False)
      self._batches_yielded += 1
    else:
      if self._mark_step_batch_count <= self._batches_yielded:
        self._batches_yielded = 0
        xm.mark_step()
      else:
        self._batches_yielded += 1

    item = self._loader.next_item(self._device)
    if item is None:
      if not self._loader._exception_queue.empty():
        raise self._loader._exception_queue.get()
      xm.mark_step()
      raise StopIteration
    return item


class ParallelLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  Args:
    cpu_loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    devices (`torch.device`...): The list of devices where the data has to be
      sent. The i-th sample returned by the `loader` will be sent to `devices[i
      % len(devices)]`.
    batchdim (int, optional): The dimension which is holding the batch size.
      Default: 0
    loader_prefetch_size (int, optional): The max capacity of the queue used by
      the thread which is reading samples from the `loader`, to be processed by
      the worker threads which upload data to the devices.
      Default: 16
    device_prefetch_size (int, optional): The max size of the per-device queues,
      where the worker threads deposit tensors which have already been sent to
      devices.
      Default: 8
    host_to_device_transfer_threads (int, optional): The number of threads that
      work in parallel to transfer data from loader queue to device queue.
      Default: 1
    input_sharding (ShardingSpec, Dict(str, ShardingSpec), optional): Sharding
      spec to apply to compatible input tensors after loading.
  """

  def __init__(self,
               cpu_loader,
               devices,
               batchdim=0,
               batches_per_execution=1,
               loader_prefetch_size=16,
               device_prefetch_size=8,
               host_to_device_transfer_threads=1,
               input_sharding=None):
    self._cpu_loader = cpu_loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._batches_per_execution = batches_per_execution
    self._done = False
    self._queues = dict()
    self._exception_queue = queue.Queue()
    self._input_sharding = input_sharding
    self._threads = []
    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    self._threads.append(thread)
    for dqueue in self._queues.values():
      for i in range(host_to_device_transfer_threads):
        thread = threading.Thread(
            target=self._worker,
            args=(
                dqueue,
                host_to_device_transfer_threads,
            ))
        thread.daemon = True
        thread.start()
        self._threads.append(thread)

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

  def per_device_samples(self):
    return len(self._cpu_loader) // len(self._devices)

  def next_item(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in self._queues.values():
      dqueue.queue.close()
      dqueue.cpu_loader_queue.close()

    for thread in self._threads:
      thread.join()

  @property
  def batches_per_execution(self):
    return self._batches_per_execution

  def _loader_worker(self):
    queues = list(self._queues.values())
    data_iter = enumerate(self._cpu_loader)
    batch = []

    try:
      while not self._done:
        try:
          with xp.Trace("cpu_loader.next"):
            _, data = next(data_iter)
        except StopIteration:
          break
        batch.append(data)
        if len(batch) == len(self._devices):
          for queue_no, device_batch in enumerate(batch):
            queues[queue_no].cpu_loader_queue.put(device_batch)
          batch = []
    finally:
      for dqueue in queues:
        dqueue.cpu_loader_queue.close_write()

  def _get_batch(self, dqueue):
    batch = []
    while len(batch) < dqueue.queue.max_size():
      item = dqueue.cpu_loader_queue.get()
      if item is None:
        break
      batch.append(item)
    return batch

  def send_cpu_data_to_device(self, batches, device):
    """Move batch to device.
    Args:
      batch -> List(torch.Tensor), List(Dict(str: torch.Tensor)): Input batch
        present in the cpu memory
      device: TPU device where the batch should be moved
    
    Returns:
      result -> List(torch.Tensor), Dict(str: torch.Tensor): Returns a dict if the
        input batch is a dict. Otherwise, returns a list of torch.Tensor.
    """
    result = None
    if isinstance(self._input_sharding, dict):
      if not isinstance(batches[0], dict):
        raise ValueError(
            f"input batch should be a dict when input sharding is a dict.")
      result = []
      for batch in batches:
        xla_batch = {}
        missing_keys = []
        for key, tensor in batch.items():
          assert type(tensor) == torch.Tensor
          sharding_spec = None
          if self._input_sharding:
            if key not in self._input_sharding:
              missing_keys.append(key)
              continue
            sharding_spec = self._input_sharding[key]

          # xla_tensor is a list of tensors.
          xla_tensor = xm.send_cpu_data_to_device(tensor, device, sharding_spec)
          xla_batch[key] = xla_tensor[0]
        if len(missing_keys) != 0:
          # Returning exception as raising in the dataloading thread doesn't surface the problem in the main thread.
          raise KeyError(
              f"Keys: {missing_keys} are missing from input_sharding.")
        result.append(xla_batch)
    else:
      result = xm.send_cpu_data_to_device(batches, device, self._input_sharding)
    return result

  def _worker(self, dqueue, host_to_device_transfer_threads):
    device = torch.device(dqueue.device)

    try:
      while True:
        with xp.Trace("get_batch_from_cpu_queue"):
          batch = self._get_batch(dqueue)
        if not batch:
          break
        with torch.no_grad():
          try:
            with xp.Trace("cpu_data_to_xla_device"):
              batch = self.send_cpu_data_to_device(batch, device)
          except Exception as e:
            # _worker is being run in a daemon thread, raise the error
            # will not work. Put the error in an error queue instead.
            self._exception_queue.put(e)
            break
        for data in batch:
          dqueue.queue.put(data)
    finally:
      close_queue_count = next(dqueue.close_queue_count)
      if close_queue_count == host_to_device_transfer_threads - 1:
        dqueue.queue.close_write()


class MpDeviceLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  This class should only be using with multi-processing data parallelism. It will wrap
  the dataloader passed in with ParallelLoader and return the per_device_loader for the
  current device.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    device (`torch.device`...): The device where the data has to be sent.
    kwargs: Named arguments for the `ParallelLoader` constructor.

  Example:

    >>> device = torch_xla.device()
    >>> train_device_loader = MpDeviceLoader(train_loader, device)
  """

  def __init__(self, loader, device, **kwargs):
    self._loader = loader
    self._device = device
    self._parallel_loader_kwargs = kwargs
    self._parallel_loader = None

  def __iter__(self):
    if self._parallel_loader is not None:
      self._parallel_loader.close()
      self._parallel_loader = None
    self._parallel_loader = ParallelLoader(self._loader, [self._device],
                                           **self._parallel_loader_kwargs)
    return self._parallel_loader.per_device_loader(self._device)

  def __len__(self):
    return len(self._loader)
