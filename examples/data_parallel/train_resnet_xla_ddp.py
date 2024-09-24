import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torchvision


class TrainResNetXLADDP(TrainResNetBase):

  def __init__(self):
    super().__init__()
    # for multiprocess we need a sampler
    train_sampler = None
    fake_dataset = xu.SampleGenerator(
        data=(torch.zeros(3, self.img_dim,
                          self.img_dim), torch.tensor(0, dtype=torch.int64)),
        sample_count=self.train_dataset_len)
    if xr.world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          fake_dataset, num_replicas=xr.world_size(), rank=xr.global_ordinal())
    train_loader = torch.utils.data.DataLoader(
        fake_dataset, batch_size=self.batch_size, sampler=train_sampler)
    self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)

  def run_optimizer(self):
    # optimizer_step will call `optimizer.step()` and all_reduce the gradident
    xm.optimizer_step(self.optimizer)


def _mp_fn(index):
  # cache init needs to happens inside the mp_fn.
  xr.initialize_cache(f'/tmp/xla_cache_{index}', readonly=False)
  xla_ddp = TrainResNetXLADDP()
  xla_ddp.start_training()


if __name__ == '__main__':
  print(
      'consider using train_resnet_spmd_data_parallel.py instead to get better performance'
  )
  #torch_xla.launch(_mp_fn, args=())
  _mp_fn(0)
