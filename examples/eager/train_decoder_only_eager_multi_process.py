import sys
import os

example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import torch_xla
import torch_xla.core.xla_model as xm


class TrainDecoderXLADDP(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    # We want to run the step fn eagerly.
    self.compiled_step_fn = self.step_fn

  def run_optimizer(self):
    # optimizer_step will call `optimizer.step()` and all_reduce the gradident
    xm.optimizer_step(self.optimizer)


def _mp_fn(index):
  import torch_xla
  torch_xla.experimental.eager_mode(True)
  xla_ddp = TrainDecoderXLADDP()
  xla_ddp.start_training()


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())
