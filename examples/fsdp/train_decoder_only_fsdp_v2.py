import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
import decoder_only_model
from train_decoder_only_base import TrainDecoderOnlyBase

import functools

import torch
import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla import runtime as xr
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

# checkout our doc at https://github.com/pytorch/xla/blob/master/docs/fsdpv2.md
class TrainDecoderOnlyFSDPv2(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    # Define the mesh following common SPMD practice
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    xs.set_global_mesh(mesh)

    # Shard the input(data parallel).
    # Scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    self.batch_size *= num_devices
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64),
              torch.zeros(self.batch_size, self.seq_len, dtype=torch.int64)),
        sample_count=self.train_dataset_len // self.batch_size)
    self.train_device_loader = pl.MpDeviceLoader(
        train_loader,
        self.device,
        # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
        input_sharding=xs.ShardingSpec(mesh, ('fsdp', None)))

    # Apply FSDP sharding on each DecoderLayer layer.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            decoder_only_model.DecoderLayer
        },
    )
    # FSDPv2 will use the global mesh set above
    self.model = FSDPv2(
        self.model, auto_wrap_policy=auto_wrap_policy)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)


if __name__ == '__main__':
  # Enable the SPMD
  xr.use_spmd()
  base = TrainDecoderOnlyFSDPv2()
  base.start_training()
