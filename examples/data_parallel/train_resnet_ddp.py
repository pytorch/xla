import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla
import torch_xla.core.xla_model as xm


class TrainResNetDDP(TrainResNetBase):

  def __init__(self):
    dist.init_process_group('xla', init_method='xla://')
    super().__init__()
    self.model = DDP(
        self.model, gradient_as_bucket_view=True, broadcast_buffers=False)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)
    # below code is commented out because in this example we used a fake data
    # loader that does not take sampler. However this logic is needed if you
    # want each process to handle different parts of the data.
    '''
    train_sampler = None
    if xr.world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xr.world_size(),
          rank=xr.global_ordinal(),
          shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers,
        persistent_workers=FLAGS.persistent_workers,
        prefetch_factor=FLAGS.prefetch_factor)
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
    '''


def _mp_fn(index):
  ddp = TrainResNetDDP()
  ddp.start_training()


if __name__ == '__main__':
  print(
      'consider using train_resnet_spmd_data_parallel.py instead to get better performance'
  )
  torch_xla.launch(_mp_fn)
