from train_resnet_base import TrainResNetBase
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch_xla.distributed.xla_multiprocessing as xmp


class TrainResNetDDP(TrainResNetBase):

  def __init__(self):
    super().__init__()
    dist.init_process_group('xla', init_method='xla://')
    self.model = DDP(
        self.model, gradient_as_bucket_view=True, broadcast_buffers=False)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)


def _mp_fn(index):
  ddp = TrainResNetDDP()
  ddp.start_training()


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
