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
        self.model, broadcast_buffers=False)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)


def _mp_fn(index):
  ddp = TrainResNetDDP()
  ddp.start_training()


if __name__ == '__main__':
  print(
      'consider using train_resnet_spmd_data_parallel.py instead to get better performance'
  )
  torch_xla.launch(_mp_fn)
