import sys
import os

example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class TrainResNetXLADDP(TrainResNetBase):

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
  torch_xla.launch(_mp_fn, args=())
