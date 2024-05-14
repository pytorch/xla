from train_resnet_base import TrainResNetBase

import numpy as np

import torch
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
from torch_xla import runtime as xr

# Enable the SPMD
xr.use_spmd()


# More detailed examaple can be found in https://github.com/pytorch/xla/blob/master/test/spmd/test_train_spmd_imagenet.py
# Check out our user guide in https://github.com/pytorch/xla/blob/master/docs/spmd.md
class TrainResNetXLASpmdDDP(TrainResNetBase):

  def __init__(self):
    super().__init__()
    # Shard along batch dimension only
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    mesh_shape = (num_devices, 1, 1, 1)
    # We know data is 4d and 0th dimension is the batch dimension
    input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
    # scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    self.batch_size *= num_devices

    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, 3, self.img_dim, self.img_dim),
              torch.zeros(self.batch_size, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)
    self.train_device_loader = pl.MpDeviceLoader(
        train_loader,
        self.device,
        input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)))


if __name__ == '__main__':
  spmd_ddp = TrainResNetXLASpmdDDP()
  spmd_ddp.start_training()
