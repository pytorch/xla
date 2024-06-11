import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import torch_xla

if __name__ == '__main__':
  torch_xla.experimental.eager_mode(True)
  base = TrainDecoderOnlyBase()
  base.start_training()
