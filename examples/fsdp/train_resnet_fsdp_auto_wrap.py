import sys
import os

example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase
from functools import partial

import torch
import torchvision
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from torch_xla.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                             transformer_auto_wrap_policy)


class TrainResNetXLAFSDP(TrainResNetBase):

  def __init__(self):
    super().__init__()
    # auto_wrap_policy can be either size_based or type_based
    auto_wrap_policy = "size_based"
    auto_wrap_min_num_params = 1e6
    if auto_wrap_policy == "size_based":
      # auto-wrap all sub-modules with a certain number of parameters (default 1e6)
      auto_wrap_policy = partial(
          size_based_auto_wrap_policy, min_num_params=auto_wrap_min_num_params)
    elif auto_wrap_policy == "type_based":
      # auto-wrap all sub-modules in torchvision ResNet's BasicBlock or Bottleneck
      # or torchvision transformer's EncoderBlock as an example
      # (transformer_auto_wrap_policy wraps all sub-modules in transformer_layer_cls)
      auto_wrap_policy = partial(
          transformer_auto_wrap_policy,
          transformer_layer_cls={
              torchvision.models.resnet.BasicBlock,
              torchvision.models.resnet.Bottleneck,
              torchvision.models.vision_transformer.EncoderBlock,
          })
    else:
      raise Exception(f"Invalid auto-wrap policy: {auto_wrap_policy}")
    self.model = FSDP(
        self.model,
        compute_dtype=torch.float32,
        pin_layout_in_collective_ops=True,
        auto_wrap_policy=auto_wrap_policy)
    self.optimizer = optim.SGD(self.model.parameters(), weight_decay=1e-4)


def _mp_fn(index):
  xla_fsdp = TrainResNetXLAFSDP()
  xla_fsdp.start_training()


if __name__ == '__main__':
  print(
      'consider using train_decoder_only_fsdp_v2.py instead to get better performance'
  )
  torch_xla.launch(_mp_fn, args=())
