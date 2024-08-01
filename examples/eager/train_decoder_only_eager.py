import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import torch_xla


class TrainDecoderOnlyEager(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    # We want to run the step fn eagerly.
    self.compiled_step_fn = self.step_fn


if __name__ == '__main__':
  torch_xla.experimental.eager_mode(True)
  base = TrainDecoderOnlyEager()
  base.start_training()
