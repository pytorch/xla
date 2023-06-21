import threading
import torch
import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.utils.keyd_queue as kq
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import tensorflow.compat.v1 as tf
import torchvision
import numpy as np

'''
ray.init()

#MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
#STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

@ray.remote
def transform_data(item, start_idx, end_idx):
  with xp.Trace('chandu'):
    transformed_batch = []
    #for idx, item in enumerate(batch):
    for i in range(start_idx, end_idx):
      (image, label) = item[start_idx: end_idx]
      #with xp.Trace('from_numpy'):
      #  image = torch.from_numpy(image)
      with xp.Trace('view'):
        image = numpy.reshape(224, 224, -1, 3)
      with xp.Trace('permute'):
        image = numpy.transpose(image, (2,3,0,1))
      with xp.Trace('bfloat16'):
        #image = torchvision.transforms.functional.convert_image_dtype(torch.bfloat16)
        image = image.to(torch.bfloat16)
      label = torch.from_numpy(label)
      item[start_idx: end_idx] = (image,label)
  return item
'''

MEAN_RGB = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(3, 1, 1)
STDDEV_RGB = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(3, 1, 1)



class PerDeviceQueue(object):

  def __init__(self, device, loader_prefetch_size, device_prefetch_size):
    self.device = device
    self.loader_queue = kq.Queue(maxsize=loader_prefetch_size)
    self.queue = kq.Queue(maxsize=device_prefetch_size)


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
      xm.mark_step()
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
    loader_prefetch_size (int, optional): The max capacity of the queue used by
      the thread which is reading samples from the `loader`, to be processed by
      the worker threads which upload data to the devices.
      Default: 8
    device_prefetch_size (int, optional): The max size of the per-device queues,
      where the worker threads deposit tensors which have already been sent to
      devices.
      Default: 4
    host_to_device_transfer_threads (int, optional): The number of threads that
      work in parallel to transfer data from loader queue to device queue.
      Default: 1
  """

  def __init__(self,
               loader,
               devices,
               batchdim=0,
               batches_per_execution=1,
               loader_prefetch_size=8,
               device_prefetch_size=4,
               host_to_device_transfer_threads=1):
    self._loader = loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._batches_per_execution = batches_per_execution
    self._done = False
    self._queues = dict()
    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
      #print(f'device: {device}')
    thread = threading.Thread(target=self._loader_worker)
    thread.daemon = True
    thread.start()
    for dqueue in self._queues.values():
      for i in range(host_to_device_transfer_threads):
        thread = threading.Thread(target=self._worker, args=(dqueue,))
        thread.daemon = True
        thread.start()
        #process = multiprocessing.Process(target=self._worker, args=(dqueue,))
        #process.start()
      

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
    return len(self._loader) // len(self._devices)

  def next_item(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in self._queues.values():
      dqueue.queue.close()
      dqueue.loader_queue.close()

  @property
  def batches_per_execution(self):
    return self._batches_per_execution

  def _loader_worker(self):
    queues = list(self._queues.values())
    data_iter = enumerate(self._loader)
    batch = []
    while not self._done:
      try:
        with xp.Trace('disk_to_device'):
          _, data = next(data_iter)
          '''
          if data is not None:
            (image, label) = data
            with xp.Trace('from_numpy'):
              image = torch.from_numpy(np.transpose(np.reshape(image, (224,224,-1,3)),(2,3,0,1)))
              label = torch.from_numpy(label)
            data = (image, label)
          '''
      except StopIteration:
        break
      batch.append(data)
      if len(batch) == len(self._devices):
        for queue_no, device_batch in enumerate(batch):
          queues[queue_no].loader_queue.put(device_batch)
        batch = []
    for dqueue in queues:
      dqueue.loader_queue.close_write()

  def _get_batch(self, dqueue):
    with xp.Trace('_get_batch'):
      batch = []
      while 4 > len(batch):
        with xp.Trace('queue_get'):
          item = dqueue.loader_queue.get()
        '''
        if item is not None:
          with xp.Trace('data_transform'):
            (image, label) = item 
            with xp.Trace('bfloat16'):
              image = image.to(torch.bfloat16)
              image -= MEAN_RGB
              image /= STDDEV_RGB
            item = (image,label)
        '''
        if item is None:
          break
        batch.append(item)
      return batch


  def _worker(self, dqueue):
    device = torch.device(dqueue.device)
    while True:
      with xp.Trace('worker1'):
        batch = self._get_batch(dqueue)
        if not batch:
          break
        with xp.Trace('send_cpu_data_to_device1'):
          batch = xm.send_cpu_data_to_device(batch, device)
        for data in batch:
          dqueue.queue.put(data)
    dqueue.queue.close_write()

  

class MpDeviceLoader(object):
  """Wraps an existing PyTorch DataLoader with background data upload.

  This class should only be using with multi-processing data parallelism.

  Args:
    loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
      wrapped.
    device (`torch.device`...): The device where the data has to be sent.
    kwargs: Named arguments for the `ParallelLoader` constructor.
  """

  def __init__(self, loader, device, **kwargs):
    self._loader = loader
    self._device = device
    self._parallel_loader_kwargs = kwargs

  def __iter__(self):
    parallel_loader = ParallelLoader(self._loader, [self._device],
                                     **self._parallel_loader_kwargs)
    return parallel_loader.per_device_loader(self._device)

  def __len__(self):
    return len(self._loader)
