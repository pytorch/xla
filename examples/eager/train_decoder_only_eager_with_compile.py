import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import torch_xla

if __name__ == '__main__':
  # The step fn will still be compiled, random input generation happens eagerly.
  torch_xla.experimental.eager_mode(True)
  trainer = TrainDecoderOnlyBase()
  trainer.start_training()
