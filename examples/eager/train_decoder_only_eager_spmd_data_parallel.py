import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

import numpy as np

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
from torch_xla import runtime as xr

# Enable the SPMD
xr.use_spmd()


# More detailed examaple can be found in https://github.com/pytorch/xla/blob/master/test/spmd/test_train_spmd_imagenet.py
# Check out our user guide in https://github.com/pytorch/xla/blob/master/docs/spmd.md
class TrainDecoderSpmdDDP(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    # Shard along batch dimension only
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    mesh_shape = (num_devices,)
    mesh = xs.Mesh(device_ids, mesh_shape, ('data',))
    # scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    self.batch_size *= num_devices

    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64),
              torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)
    self.train_device_loader = pl.MpDeviceLoader(
        train_loader,
        self.device,
        # Shard the input's batch dimension along the `data` axis, no sharding along other dimensions
        input_sharding=xs.ShardingSpec(mesh, ('data', None)))


if __name__ == '__main__':
  torch_xla.experimental.eager_mode(True)
  spmd_ddp = TrainDecoderSpmdDDP()
  spmd_ddp.start_training()