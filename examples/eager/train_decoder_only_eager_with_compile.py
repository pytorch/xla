import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import torch_xla


class TrainDecoderOnlyEagerWithCompile(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    # step fn will be compiled and rest will be run eagerly.
    self.step_fn = torch_xla.experimental.compile(self.step_fn)


if __name__ == '__main__':
  torch_xla.experimental.eager_mode(True)
  trainer = TrainDecoderOnlyEagerWithCompile()
  trainer.start_training()
